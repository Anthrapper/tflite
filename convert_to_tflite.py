import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

import tensorflow_hub as hub

# model with Sequential api

# universal-sentence-encoder layer
# directly from tfhub
#centralized
# base = tf.keras.Sequential(
#     [tf.keras.layers.Embedding( 
#         input_dim=10000,
#         output_dim=100,
#         input_length=200), 
#      tf.keras.layers.Lambda(lambda x: x)
#     ]
# )
# base = tf.keras.Sequential(
#     [tf.keras.layers.Embedding( 
#         input_dim=500,
#         output_dim=16,
#         input_length=50, 
#         dtype='float32',
#     ), 
#      tf.keras.layers.Lambda(lambda x: x)
#     ]
# )

base = tf.keras.Sequential(
    [
     layers.Input(shape=(120,120,3)),   
     tf.keras.layers.Lambda(lambda x: x)
    ]
)

base.compile(loss="binary_crossentropy", optimizer="adam")
base.save("identity_model", save_format="tf")

head = tf.keras.Sequential(
    [           
    layers.Input(shape=(120,120,3)),
    layers.Conv2D(10, (3,3), activation="relu", input_shape=(120,120,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
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
