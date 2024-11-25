import keras
from keras import layers


def _create_conv_relu(
    inputs, name, filters, dropout_ratio, is_training, strides=(1, 1), kernel_size=(3, 3), padding="same", relu=True
):
    """
    Create a convolutional block with optional ReLU, dropout, and batch normalization
    """
    conv = layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f"{name}_conv"
    )(inputs)

    if dropout_ratio > 0:
        conv = layers.Dropout(rate=dropout_ratio, name=f"{name}_dropout")(conv, training=is_training)

    conv = layers.BatchNormalization(center=True, scale=False, name=f"{name}_bn")(conv, training=is_training)

    if relu:
        conv = layers.ReLU(name=f"{name}_relu")(conv)

    return conv


def create_unet2(num_layers, num_filters, data, is_training, prev=None, dropout_ratio=0, classes=3):
    """
    Create U-Net architecture with consistent layer naming for weight loading
    """
    # Contracting path
    skips = []
    x = data

    # Down blocks
    for i in range(num_layers):
        x = _create_conv_relu(x, f"c_{i}_1", num_filters * (2**i), dropout_ratio=dropout_ratio, is_training=is_training)
        x = _create_conv_relu(x, f"c_{i}_2", num_filters * (2**i), dropout_ratio=dropout_ratio, is_training=is_training)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name=f"pool_{i}")(x)

    # Middle blocks
    middle_filters = num_filters * (2**num_layers)
    x = _create_conv_relu(x, "m_1", middle_filters, dropout_ratio=dropout_ratio, is_training=is_training)
    x = _create_conv_relu(x, "m_2", middle_filters, dropout_ratio=dropout_ratio, is_training=is_training)

    # Up blocks
    for i in range(num_layers - 1, -1, -1):
        current_filters = num_filters * (2**i)
        x = layers.Conv2DTranspose(
            filters=current_filters, kernel_size=(2, 2), strides=(2, 2), padding="same", name=f"e_{i}_upconv"
        )(x)
        x = layers.Concatenate(axis=3, name=f"concat_e_{i}")([skips[i], x])
        x = _create_conv_relu(x, f"e_{i}_1", current_filters, dropout_ratio=dropout_ratio, is_training=is_training)
        x = _create_conv_relu(x, f"e_{i}_2", current_filters, dropout_ratio=dropout_ratio, is_training=is_training)

    last_relu = x

    if prev is not None:
        x = layers.Concatenate(axis=3, name="final_concat")([prev, x])

    # Final layers
    conv_logits = _create_conv_relu(x, "conv_logits", num_filters, dropout_ratio=dropout_ratio, is_training=is_training)
    logits = _create_conv_relu(conv_logits, "logits", classes, dropout_ratio=dropout_ratio, is_training=is_training)

    # Angle prediction branch
    conv_angle = _create_conv_relu(
        x, "conv_angle", num_filters, dropout_ratio=dropout_ratio, is_training=is_training, relu=False
    )
    angle_pred = _create_conv_relu(
        conv_angle, "angle_pred", 1, dropout_ratio=dropout_ratio, is_training=is_training, relu=False
    )

    return logits, last_relu, angle_pred
