# find_det.py

import os
import tensorflow as tf
from keras import layers, Model
import time
import numpy as np
from tqdm import tqdm
import math
import shutil
import itertools
from utils.paths import DATA_DIR, IMG_DIR, POS_DIR, TMP_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES
from utils import func
from . import unet
from . import segm_proc

BATCH_SIZE = 4


def read_all_files():
    drs = [""]
    fls = []
    for dr in drs:
        dr_fls = os.listdir(IMG_DIR)
        dr_fls.sort()
        fls.extend(map(lambda fl: os.path.join(dr, fl), dr_fls))
    print(f"{len(fls)} files", flush=True)
    return fls


def generate_offsets_for_frame():
    xs = range(0, func.FR_D, DS)
    ys = range(0, func.FR_D, DS)
    return list(itertools.product(xs, ys))


def process_detections(output, offs, cur_fr):
    res = np.zeros((0, 4))
    for batch_i in range(BATCH_SIZE):
        (off_x, off_y) = offs[batch_i]
        if (off_x >= 0) and (off_y >= 0):
            prs = segm_proc.extract_positions(output[batch_i, 0, :, :], output[batch_i, 1, :, :])
            res_batch = np.zeros((len(prs), 4))
            for i in range(len(prs)):
                (x, y, cl, a, ax) = prs[i]
                ax_d = math.degrees(ax)
                a_d = math.degrees(a)
                ax_d = ax_d + 180 if (segm_proc.angle_diff(a_d, ax_d) > 90) else ax_d
                res_batch[i, :] = [x, y, cl, ax_d]
            res_batch[:, 0] += off_x
            res_batch[:, 1] += off_y
            res = np.append(res, res_batch, axis=0)

    output_file = os.path.join(POS_DIR, f"{cur_fr:06d}.txt")
    with open(output_file, "a") as f:
        np.savetxt(f, res, fmt="%i", delimiter=",", newline="\n")
    return len(res)


class DetectionInference:

    def __init__(self):
        # Initialize with correct channel dimensions
        self.batch_data = np.zeros((BATCH_SIZE, DS, DS, 1), dtype=np.float32)
        self.prior_data = np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None

    def load_tf1_weights(self, checkpoint_prefix):
        """Load weights from TF1 checkpoint."""
        weight_dict = {}
        reader = tf.train.load_checkpoint(checkpoint_prefix)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for key in var_to_shape_map:
            if "Adam" in key or "beta" in key or "moving" in key:
                continue
            weight = reader.get_tensor(key)
            clean_name = key.replace("/", "_")
            clean_name = clean_name.replace("kernel", "weight")
            clean_name = clean_name.replace("gamma", "weight")
            clean_name = clean_name.replace("beta", "bias")
            weight_dict[clean_name] = weight

        return weight_dict

    def build_model(self, checkpoint_dir):
        self.is_train = False

        # Define inputs with correct shapes
        self.input_img = layers.Input(shape=(DS, DS, 1), batch_size=BATCH_SIZE, name="images")
        self.input_prior = layers.Input(shape=(DS, DS, NUM_FILTERS), batch_size=BATCH_SIZE, name="prior")

        # Get outputs directly using create_unet2
        logits, last_relu, angle_pred = unet.create_unet2(
            NUM_LAYERS, 
            NUM_FILTERS, 
            self.input_img, 
            self.is_train, 
            prev=self.input_prior, 
            classes=CLASSES
        )

        self.model = Model(
            inputs={"images": self.input_img, "prior": self.input_prior},
            outputs=[logits, angle_pred, last_relu]
        )
        # Find latest checkpoint number
        checkpoint_nb = func.find_last_checkpoint(checkpoint_dir)
        # TODO: Remove later
        checkpoint_nb = 20

        # Construct checkpoint prefix (without .index or .data-* suffix)
        checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt-{checkpoint_nb}")

        print(f"Restoring checkpoint {checkpoint_nb} from {checkpoint_prefix}..", flush=True)

        try:
            # Verify checkpoint files exist
            if not os.path.exists(f"{checkpoint_prefix}.index"):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_prefix}.index")

            # Try loading the checkpoint directly
            self.model.load_weights(checkpoint_prefix)
            print("Successfully loaded checkpoint weights")

        except (ValueError, tf.errors.NotFoundError) as e:
            print(f"Error loading checkpoint directly: {e}")
            print("Attempting to convert TF1 checkpoint to Keras format...")

            try:
                weight_dict = self.load_tf1_weights(checkpoint_prefix)

                for layer in self.model.layers:
                    layer_name = layer.name
                    if layer_name in weight_dict:
                        layer.set_weights([weight_dict[layer_name]])

                # Save in Keras format
                keras_checkpoint_file = os.path.join(checkpoint_dir, f"ckpt-{checkpoint_nb:06d}.weights.h5")
                self.model.save_weights(keras_checkpoint_file)
                print(f"Saved converted weights to {keras_checkpoint_file}")

            except Exception as e:
                print(f"Error converting checkpoint: {e}")
                print("\nExpected checkpoint files:")
                print(f"- {checkpoint_prefix}.index")
                print(f"- {checkpoint_prefix}.data-00000-of-00001")
                raise

    def _feed_dict(self, offs, cur_fr, priors):
        img = func.read_img(cur_fr, IMG_DIR)
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):
                self.batch_data[batch_i, :, :, 0] = img[off_y : (off_y + DS), off_x : (off_x + DS)]
            else:
                self.batch_data[batch_i, :, :, :] = 0

        # Ensure priors have correct shape
        if priors.shape != (BATCH_SIZE, DS, DS, NUM_FILTERS):
            raise ValueError(f"Prior shape mismatch. Expected {(BATCH_SIZE, DS, DS, NUM_FILTERS)}, got {priors.shape}")

        return {"images": self.batch_data.astype(np.float32), "prior": priors.astype(np.float32)}

    def _load_offs_for_run(self, offsets, start_i):
        res = []
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = (-1, -1) if start_i >= len(offsets) else offsets[start_i]
            res.append((off_x, off_y))
            start_i = start_i + 1
        return res, start_i


    def run_inference(self, fls, offsets, start_off_i=0):
        t1 = time.time()
        output_i = 0
        n_runs = math.ceil(len(offsets) / BATCH_SIZE)
        total_bees = 0

        with tqdm(total=n_runs * len(fls), desc="Processing", unit="patches") as pbar:
            for i in range(n_runs):
                run_offs, start_off_i = self._load_offs_for_run(offsets, start_off_i)
                last_priors = np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32)

                for cur_fr in range(len(fls)):
                    inputs = self._feed_dict(run_offs, cur_fr, last_priors)
                    outs = self.model.predict(inputs, verbose=0, batch_size=BATCH_SIZE)

                    # Process outputs
                    log_res = tf.argmax(outs[0], axis=3)
                    if hasattr(log_res, "numpy"):
                        log_res = log_res.numpy()
                    angle_res = outs[1][:, :, :, 0]

                    # Combine results
                    detection_output = np.stack([log_res, angle_res], axis=1)

                    # Process detections
                    num_bees = process_detections(detection_output, run_offs, cur_fr)
                    total_bees += num_bees

                    # Update priors for next iteration - ensure correct shape
                    last_priors = outs[2]  # This should now have the correct shape

                    output_i += 1
                    pbar.update(1)
                    pbar.set_postfix({"frame": f"{cur_fr + 1}/{len(fls)}", "run": f"{i + 1}/{n_runs}", "bees": num_bees})

        elapsed_mins = (time.time() - t1) / 60
        print(f"\nProcessing complete!")
        print(f"Total time: {elapsed_mins:.1f} minutes")
        print(f"Total bees detected: {total_bees}")
        print(f"Average processing time per frame: {(elapsed_mins * 60) / len(fls):.1f} seconds")


def find_detections(checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2")):
    print(DATA_DIR)
    if os.path.exists(POS_DIR):
        shutil.rmtree(POS_DIR)
    os.mkdir(POS_DIR)

    fls = read_all_files()
    offsets = generate_offsets_for_frame()

    with DetectionInference() as model_obj:
        model_obj.build_model(checkpoint_dir)
        model_obj.run_inference(fls, offsets)

    # Verify results
    result_files = os.listdir(POS_DIR)
    print(f"Detection files created: {len(result_files)}")
