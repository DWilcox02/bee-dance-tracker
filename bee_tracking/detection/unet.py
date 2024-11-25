import tensorflow as tf
from keras import layers
import keras


def _create_conv_relu(
    inputs, name, filters, dropout_ratio, is_training, strides=(1, 1), kernel_size=(3, 3), padding="same", relu=True
):
    """
    Create a convolutional block with optional ReLU, dropout, and batch normalization
    """
    # Create a sequential block for better organization
    conv_block = keras.Sequential(name=name)

    # Add convolution layer
    conv_block.add(
        layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f"{name}_conv")
    )

    # Add dropout if specified
    if dropout_ratio > 0:
        conv_block.add(layers.Dropout(rate=dropout_ratio, name=f"{name}_dropout"))

    # Add batch normalization
    conv_block.add(layers.BatchNormalization(center=True, scale=False, name=f"{name}_bn"))

    # Add ReLU if specified
    if relu:
        conv_block.add(layers.ReLU(name=f"{name}_relu"))

    return conv_block(inputs, training=is_training)


class ResizeLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, target_shape=None):
        return tf.image.resize(inputs, target_shape, method="nearest")

class ConcatLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs1, inputs2):
        return tf.concat([inputs1, inputs2], axis=-1)


class ConcatResizeLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.resize = ResizeLayer()  # Use the flexible ResizeLayer instead of fixed-size Resizing

    def call(self, inputs, prev):
        # Resize prev to match inputs shape
        resized = self.resize(inputs=prev, target_shape=tf.shape(inputs)[1:3])
        # Concatenate
        return self.concat([inputs, resized])


class ConvBlock(keras.layers.Layer):
    def __init__(self, filters, name, dropout_ratio=0, relu=True):
        super().__init__(name=name)
        self.conv = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", name=f"{name}_conv")
        self.dropout = layers.Dropout(rate=dropout_ratio, name=f"{name}_dropout") if dropout_ratio > 0 else None
        self.batch_norm = layers.BatchNormalization(center=True, scale=False, name=f"{name}_bn")
        self.relu = layers.ReLU(name=f"{name}_relu") if relu else None

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.dropout:
            x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)
        if self.relu:
            x = self.relu(x)
        return x


class UNet(tf.keras.Model):

    def __init__(self, num_layers, num_filters, dropout_ratio=0, classes=2):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.dropout_ratio = dropout_ratio
        self.classes = classes

        # Initial conv layer to handle concatenated input (1 + NUM_FILTERS channels)
        self.initial_conv = layers.Conv2D(num_filters, 3, padding="same", activation="relu")

        # Encoder path
        self.encoder_convs = []
        self.encoder_pools = []
        for i in range(num_layers):
            n_filters = num_filters * (2**i)
            conv = tf.keras.Sequential(
                [
                    layers.Conv2D(n_filters, 3, padding="same", activation="relu"),
                    layers.Conv2D(n_filters, 3, padding="same", activation="relu"),
                ]
            )
            self.encoder_convs.append(conv)
            self.encoder_pools.append(layers.MaxPooling2D(pool_size=(2, 2)))

        # Bottom - use num_filters instead of multiplying it
        self.bottom = tf.keras.Sequential(
            [
                layers.Conv2D(num_filters, 3, padding="same", activation="relu"),
                layers.Conv2D(num_filters, 3, padding="same", activation="relu"),
            ]
        )

        # Decoder path
        self.decoder_upconvs = []
        self.decoder_convs = []
        for i in range(num_layers - 1, -1, -1):
            n_filters = num_filters * (2**i)
            upconv = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")
            self.decoder_upconvs.append(upconv)

            conv = tf.keras.Sequential(
                [
                    layers.Conv2D(n_filters, 3, padding="same", activation="relu"),
                    layers.Conv2D(n_filters, 3, padding="same", activation="relu"),
                ]
            )
            self.decoder_convs.append(conv)

        # Final layers
        self.final_conv = layers.Conv2D(classes, 1, padding="same")
        self.angle_conv = layers.Conv2D(1, 1, padding="same")

        # Additional layers
        self.concat = layers.Concatenate(axis=-1)
        self.resize = ResizeLayer()

        if dropout_ratio > 0:
            self.dropout = layers.Dropout(dropout_ratio)

    def call(self, inputs, training=False):
        # Initial conv to handle concatenated input
        x = self.initial_conv(inputs)
        initial_shape = tf.shape(inputs)[1:3]  # Store initial shape

        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder path
        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            x = conv(x)
            encoder_outputs.append(x)
            x = pool(x)

        # Bottom
        x = self.bottom(x)
        last_relu = x  # Store for returning later

        # Decoder path
        for i, (upconv, conv) in enumerate(zip(self.decoder_upconvs, self.decoder_convs)):
            x = upconv(x)

            # Get corresponding encoder output
            encoder_output = encoder_outputs[-(i + 1)]

            # Resize if dimensions don't match
            if x.shape[1:3] != encoder_output.shape[1:3]:
                x = self.resize(inputs=x, target_shape=encoder_output.shape[1:3])

            # Concatenate
            x = self.concat([encoder_output, x])

            x = conv(x)

            if self.dropout_ratio > 0 and training:
                x = self.dropout(x)

        # Ensure last_relu is resized to match input dimensions
        last_relu = self.resize(inputs=last_relu, target_shape=initial_shape)

        # Final layers
        logits = self.final_conv(x)
        angle_pred = self.angle_conv(x)

        return logits, last_relu, angle_pred


class UNetModel(tf.keras.Model):

    def __init__(self, num_layers, num_filters, classes=2):
        super().__init__()
        self.concat_resize = ConcatResizeLayer()
        self.unet = UNet(num_layers, num_filters, classes=classes)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            x = inputs["images"]
            prev = inputs["prior"]

            # Use concat-resize layer to handle the inputs
            x = self.concat_resize(x, prev)
        else:
            x = inputs

        return self.unet(x, training=training)


def create_unet2(num_layers, num_filters, inputs, is_training, prev=None, classes=2):
    """Factory function to create a U-Net model."""
    if prev is not None:
        # Use the concat-resize layer
        concat_layer = ConcatResizeLayer()
        combined_inputs = concat_layer(inputs, prev)
    else:
        combined_inputs = inputs

    model = UNet(num_layers, num_filters, dropout_ratio=0 if not is_training else 0.5, classes=classes)
    return model(combined_inputs, training=is_training)
