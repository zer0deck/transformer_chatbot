"""Sequential model file
"""
import json
from dataclasses import dataclass, field
import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
from chatbot.preprocessor import Corpus

