import tensorflow as tf
from tensorflow.keras import layers
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter



base = tf.keras.Sequential(
    [
     layers.Input(shape=(100,)),   
     tf.keras.layers.Lambda(lambda x: x)
    ]
)

base.compile(loss="binary_crossentropy", optimizer="adam")
base.save("identity_model", save_format="tf")

head = tf.keras.Sequential(
    [           
    layers.Input(shape=(100,)),   
    layers.Embedding(input_dim=5965,output_dim=100),
    layers.LSTM(32,return_sequences = True),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dropout(rate=0.3),
    layers.Dense(1, activation="sigmoid")
    ]
)

head.compile(loss="binary_crossentropy", optimizer="adam")


"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""

base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    2, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-3), train_batch_size=32
)

converter.convert_and_save("tflite_model")
