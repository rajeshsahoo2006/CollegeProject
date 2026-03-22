#!/usr/bin/env python3
"""Run CNN stride comparison - trains both models and prints results."""
import os
os.environ["MPLBACKEND"] = "Agg"  # Non-interactive backend

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

def build_model_stride2():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, strides=2, activation="relu", padding="same"),
        layers.Conv2D(64, 3, strides=2, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

def build_model_stride3():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, strides=3, activation="relu", padding="same"),
        layers.Conv2D(64, 3, strides=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

EPOCHS = 5
BATCH_SIZE = 128

print("\n" + "=" * 55)
print("Training Model 1 (Stride=2 - Original)")
print("=" * 55)
model_stride2 = build_model_stride2()
model_stride2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history2 = model_stride2.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

print("\n" + "=" * 55)
print("Training Model 2 (Stride=3 - Custom)")
print("=" * 55)
model_stride3 = build_model_stride3()
model_stride3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history3 = model_stride3.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

print("\n" + "=" * 55)
print("COMPARISON - Test Set Evaluation")
print("=" * 55)
loss2, acc2 = model_stride2.evaluate(x_test, y_test, verbose=0)
loss3, acc3 = model_stride3.evaluate(x_test, y_test, verbose=0)

print(f"\nModel 1 (Stride=2):  Test Loss = {loss2:.4f}  |  Test Accuracy = {acc2:.4f} ({acc2*100:.2f}%)")
print(f"Model 2 (Stride=3):  Test Loss = {loss3:.4f}  |  Test Accuracy = {acc3:.4f} ({acc3*100:.2f}%)")
print(f"\nAccuracy difference: {acc2 - acc3:+.4f} (Stride=2 {'wins' if acc2 > acc3 else 'loses'})")
print(f"Final val accuracy - Stride=2: {history2.history['val_accuracy'][-1]:.4f}, Stride=3: {history3.history['val_accuracy'][-1]:.4f}")
