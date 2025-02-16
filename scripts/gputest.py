import tensorflow as tf 

"""
This script is used to test if the GPU is available for use. 
"""

print(tf.config.list_physical_devices('GPU'))