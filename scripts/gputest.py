"""
This script is used to test if the GPU is available for use.
"""

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
