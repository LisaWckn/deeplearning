# Source - https://stackoverflow.com/a/63421166
# Posted by Gerry P
# Retrieved 2026-05-08, License - CC BY-SA 4.0

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU:", tf.config.list_physical_devices('GPU'))
