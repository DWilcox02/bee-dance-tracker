# train_detection.py

import os
import re
import time
import math
from random import shuffle, randint
import tensorflow as tf
from tqdm import tqdm
import keras
import numpy as np
from . import unet
from utils import func
from utils.paths import DET_DATA_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES
from plots import segm_map

BATCH_SIZE = 4
BASE_LR = 0.0001


def flip_h(data):
    data = np.flip(data, axis=2)

    for fr in range(data.shape[0]):
        ang = data[fr, 2, :, :]
        is_1 = data[fr, 1, :, :] == 1
        ang[is_1] = ((math.pi - ang[is_1] * 2 * math.pi) % (2 * math.pi)) / (2 * math.pi)
        data[fr, 2, :, :] = ang
    return data


def flip_v(data):
    data = np.flip(data, axis=3)

    for fr in range(data.shape[0]):
        ang = data[fr, 2, :, :]
        is_1 = data[fr, 1, :, :] == 1
        ang[is_1] = 1 - ang[is_1]
        data[fr, 2, :, :] = ang
    return data


class TrainModel(keras.Model):

    def __init__(self, data_path, train_prop, with_augmentation, dropout_ratio=0):
        self.data_path = data_path
        self.input_files = [f for f in os.listdir(data_path) if re.search("npz", f)]
        shuffle(self.input_files)
        self.train_prop = train_prop
        self.with_augmentation = with_augmentation
        self.dropout_ratio = dropout_ratio

        # Create the U-Net model - note the changed parameter name from num_classes to classes
        self.unet_model = unet.UNet(
            num_layers=NUM_LAYERS,
            num_filters=NUM_FILTERS,
            dropout_ratio=dropout_ratio,
            classes=CLASSES,  # Changed from num_classes to classes to match UNet's __init__
        )

        # Initialize optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=BASE_LR)

        # Initialize metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.bg_overlap_metric = keras.metrics.Mean(name="bg_overlap")
        self.fg_overlap_metric = keras.metrics.Mean(name="fg_overlap")
        self.class_error_metric = keras.metrics.Mean(name="class_error")
        self.angle_error_metric = keras.metrics.Mean(name="angle_error")

        # Initialize checkpoint manager
        self.checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_manager = None  # Will be set in build_model

    def build_model(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Setup checkpoint manager (keeping this just for saving, not loading)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=5)

        # Remove checkpoint restoration and always start from 0
        return 0

    @tf.function
    def train_step(self, inputs):
        images, labels, weights, angle_labels = inputs

        with tf.GradientTape() as tape:
            # Forward pass
            logits, last_relu, angle_pred = self.unet_model(images, training=True)

            # Ensure predictions match target dimensions
            if logits.shape[1:3] != labels.shape[1:3]:
                logits = tf.image.resize(logits, labels.shape[1:3])
                angle_pred = tf.image.resize(angle_pred, labels.shape[1:3])

            loss_softmax = self.compute_loss(labels, logits, weights)
            loss_angle = self.compute_angle_loss(angle_labels, angle_pred[:, :, :, 0], weights)
            total_loss = loss_softmax + loss_angle

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.unet_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet_model.trainable_variables))

        return logits, total_loss, last_relu, angle_pred

    @tf.function
    def test_step(self, inputs):
        images, labels, weights, angle_labels = inputs

        # Forward pass
        logits, last_relu, angle_pred = self.unet_model(images, training=False)

        loss_softmax = self.compute_loss(labels, logits, weights)
        loss_angle = self.compute_angle_loss(angle_labels, angle_pred[:, :, :, 0], weights)
        total_loss = loss_softmax + loss_angle

        return logits, total_loss, last_relu, angle_pred

    def compute_loss(self, labels, logits, weights):
        """Calculate weighted cross entropy loss."""
        # Ensure shapes match
        if logits.shape[1:3] != labels.shape[1:3]:
            logits = tf.image.resize(logits, labels.shape[1:3])

        loss = tf.cast(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True), tf.float32)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    def compute_angle_loss(self, true_angles, pred_angles, weights):
        """Calculate weighted MSE loss for angle prediction."""
        # Ensure shapes match
        pred_angles = tf.cast(pred_angles, tf.float32)
        if pred_angles.shape[1:3] != true_angles.shape[1:3]:
            pred_angles = tf.image.resize(pred_angles, true_angles.shape[1:3])
        loss = tf.reduce_mean(tf.square(pred_angles - true_angles), axis=-1)

        # Adjust weights to match the shape of loss

        weights = tf.reduce_mean(weights, axis=-1)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    def _sample_offsets(self, data):
        res = np.zeros((BATCH_SIZE, data.shape[0], data.shape[1], DS, DS))
        for i in range(BATCH_SIZE):
            off_x, off_y, fh, fv = (
                randint(0, data.shape[2] - DS),
                randint(0, data.shape[3] - DS),
                randint(0, 1),
                randint(0, 1),
            )
            if not self.with_augmentation:
                fh, fv = 0, 0
            cut_data = np.copy(data[:, :, off_x : (off_x + DS), off_y : (off_y + DS)])
            if fh:
                cut_data = flip_h(cut_data)
            if fv:
                cut_data = flip_v(cut_data)
            res[i] = cut_data
        return res, np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32)

    def _prepare_batch(self, step, batch_data):
        """Prepare batch with correct dimensions."""
        return (
            tf.cast(tf.reshape(batch_data[:, step, 0, :, :], (BATCH_SIZE, DS, DS, 1)), tf.float32),
            tf.cast(batch_data[:, step, 1, :, :], tf.int32),
            tf.cast(batch_data[:, step, 3, :, :], tf.float32),
            tf.cast(batch_data[:, step, 2, :, :], tf.float32),
        )

    def _calculate_metrics(self, step, loss, logits, angle_preds, batch_data):
        batch_data = batch_data[:, step, :, :, :]

        pred_class = tf.argmax(logits, axis=3)
        pred_angle = angle_preds[:, :, :, 0]

        lb = batch_data[:, 1, :, :]
        angle = batch_data[:, 2, :, :]
        is_bg = lb == 0
        is_fg = ~is_bg
        n_fg = tf.reduce_sum(tf.cast(is_fg, tf.float32))

        bg = tf.reduce_sum(tf.cast((pred_class[is_bg] == 0) & (pred_angle[is_bg] < 0), tf.float32)) / tf.reduce_sum(
            tf.cast(is_bg, tf.float32)
        )

        fg = 0.0
        fg_err = tf.reduce_max(tf.cast(lb, tf.float32))
        angle_err = 0.0

        if n_fg > 0:
            fg = tf.reduce_sum(tf.cast(pred_class[is_fg] != 0, tf.float32)) / n_fg
            fg_err = tf.reduce_mean(tf.cast(lb[is_fg] != pred_class[is_fg], tf.float32))
            angle_err = tf.reduce_mean(tf.abs(pred_angle[is_fg] - angle[is_fg]))

        return [0.0, loss, bg, fg, fg_err, angle_err]

    def run_train_test_iter(self, itr, plot=False):
        file = self.input_files[itr % len(self.input_files)]
        npz = np.load(os.path.join(self.data_path, file))
        data = npz["data"]
        t1 = time.time()

        train_steps = int(data.shape[0] * self.train_prop)
        batch_data, _ = self._sample_offsets(data)

        # Training loop
        accuracy_t = np.zeros((6))
        for step in range(train_steps):
            batch = self._prepare_batch(step, batch_data)
            logits, loss, _, angle_pred = self.train_step(batch)
            accuracy_t += self._calculate_metrics(step, loss, logits, angle_pred, batch_data)

        accuracy_t = accuracy_t / train_steps
        accuracy_t[0] = 0

        # Log training metrics
        tqdm.write(
            f"TRAIN - time: {(time.time() - t1) / 60:.3f} min, "
            f"loss: {accuracy_t[1]:.3f}, "
            f"background overlap: {accuracy_t[2]:.3f}, "
            f"foreground overlap: {accuracy_t[3]:.3f}, "
            f"class error: {accuracy_t[4]:.3f}, "
            f"angle error: {accuracy_t[5]:.3f}",
            nolock=False,
        )

        # Save metrics
        np.savetxt(
            os.path.join(self.checkpoint_dir, "accuracy.csv"),
            accuracy_t.reshape(1, -1),
            fmt="%.5f",
            delimiter=",",
            newline="\n",
        )

        # Testing phase
        img = []
        if step < data.shape[0]:
            img = self.run_test(batch_data, train_steps, plot)

        return img

    def run_test(self, batch_data, last_step, plot=False):
        t1 = time.time()
        res_img = []
        accuracy_t = np.zeros((6))

        for step in range(last_step, batch_data.shape[1]):
            batch = self._prepare_batch(step, batch_data)
            logits, loss, _, angle_pred = self.test_step(batch)
            tqdm.write(f"Step: {step}, Loss: {loss:.3f}", nolock=False)
            accuracy_t += self._calculate_metrics(step, loss, logits, angle_pred, batch_data)

            if step == (batch_data.shape[1] - 1) and plot:
                for i in range(BATCH_SIZE):
                    im_segm = segm_map.plot_segm_map_np(batch_data[i, step, 0, :, :], tf.argmax(logits[i], axis=2))
                    im_angle = segm_map.plot_angle_map_np(batch_data[i, step, 0, :, :], angle_pred[i])
                    res_img.append((im_segm, im_angle))

        accuracy_t = accuracy_t / (batch_data.shape[1] - last_step)
        accuracy_t[0] = 1

        tqdm.write(
            f"TEST - time: {(time.time() - t1) / 60:.3f} min, "
            f"loss: {accuracy_t[1]:.3f}, "
            f"background overlap: {accuracy_t[2]:.3f}, "
            f"foreground overlap: {accuracy_t[3]:.3f}, "
            f"class error: {accuracy_t[4]:.3f}, "
            f"angle error: {accuracy_t[5]:.3f}", 
            nolock=False
        )

        return res_img


def run_training(
    data_path=DET_DATA_DIR,
    checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2"),
    train_prop=0.9,
    n_iters=20,
    with_augmentation=True,
    return_img=False,
):

    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Create and compile model
    model = TrainModel(data_path, train_prop, with_augmentation)

    # Build model and get starting iteration
    start_iter = model.build_model(checkpoint_dir)

    # Training loop with overall progress bar
    main_pbar = tqdm(range(start_iter, start_iter + n_iters), desc="Overall Progress", leave=True)
    for i in main_pbar:
        # tqdm.write(f"\nITERATION: {i}")
        img = model.run_train_test_iter(i, plot=return_img)
        model.checkpoint_manager.save()
        main_pbar.set_postfix({"iteration": i})

    return model, img, start_iter + n_iters
