import sys

import keras
import numpy as np
import sklearn
import streamlit
import tensorflow as tf

print("Python      :", sys.version.split()[0])
print("TensorFlow  :", tf.__version__)
print("Keras       :", keras.__version__)
print("NumPy       :", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("Streamlit   :", streamlit.__version__)
print("GPUs found  :", tf.config.list_physical_devices("GPU"))

# Tiny model test: confirms TensorFlow can actually train
x = np.random.rand(64, 4).astype("float32")
y = np.random.randint(0, 2, size=(64,))
model = keras.Sequential(
    [
        keras.Input(shape=(4,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x, y, epochs=2, verbose=0)
print("Mini training test: OK")