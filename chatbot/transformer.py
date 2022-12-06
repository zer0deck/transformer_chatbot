import os
import sys
import re
import time
import tensorflow as tf
import tensorflow_datasets as tfds


assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
