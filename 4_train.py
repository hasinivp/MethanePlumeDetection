"""
Stage 1 - File 4: Train the ConvLSTM heatmap model.

Input : X_train/X_val (N,T,11,11,3), y_train/y_val (N,11,11)
Output: stage1_model.h5

The model takes a T-day sequence of [ch4,u,v] patches and predicts an
11x11 CH4-anomaly heatmap. No Lambda layers are used, so the saved .h5
reloads cleanly for inference in file 5.
"""

import numpy as np
import tensorflow as tf


def build_model(n_timesteps):
    inputs = tf.keras.Input(shape=(n_timesteps, 11, 11, 3))

    x = tf.keras.layers.ConvLSTM2D(
        32, (3, 3), padding="same", return_sequences=True, activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ConvLSTM2D(
        64, (3, 3), padding="same", return_sequences=False, activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def main():
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)

    n_timesteps = X_train.shape[1]
    model = build_model(n_timesteps)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    # model outputs (11,11,1); add channel dim to labels
    y_train_r = np.expand_dims(y_train, axis=-1)
    y_val_r = np.expand_dims(y_val, axis=-1)

    model.fit(
        X_train, y_train_r,
        validation_data=(X_val, y_val_r),
        epochs=30,
        batch_size=16,
        shuffle=True,
        verbose=1,
    )

    loss, mae = model.evaluate(X_val, y_val_r, verbose=0)
    print(f"\nValidation  loss(MSE)={loss:.5f}  MAE={mae:.5f}")

    model.save("stage1_model.h5")
    print("Saved stage1_model.h5")


if __name__ == "__main__":
    main()
