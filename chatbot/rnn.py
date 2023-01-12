# pylint: disable=[line-too-long, protected-access, arguments-differ]
"""Bidirectional RNN model file
"""
import json
import re
from dataclasses import dataclass, field
import tensorflow as tf
import pandas as pd
import numpy as np
from chatbot.preprocessor import Corpus

@dataclass(frozen=False, kw_only=True, slots=True)
class Recurrent():
    """Bidirectional model
    """
    num_epoch: int = field(default=1, init=True, repr=True)
    lang: str = field(default="en-us", init=True, repr=True)
    max_length: int = field(default=40, init=True, repr=True)
    batch_size: int = field(default=64, init=True, repr=True)
    buffer_size: int = field(default=20000, init=True, repr=True)
    optimizer: str = field(default='SGD', init=True, repr=True)
    learning_rate: float = field(default=0.1, init=True, repr=True)
    _optimizer: tf.keras.optimizers.SGD = field(default=None, init=False, repr=False)
    _beta_1: float = field(default=0.9, init=True, repr=False)
    _beta_2: float = field(default=0.98, init=True, repr=False)
    _epsilon: float = field(default=1e-9, init=True, repr=False)
    _momentum: float = field(default=0.9, init=True, repr=False)
    _data_controller: Corpus = field(default_factory=Corpus, init=False, repr=False)
    _model: tf.keras.Model = field(default_factory=tf.keras.Model, init=False, repr=False)

    def __post_init__(self):
        tf.keras.backend.clear_session()
        assert self.optimizer == 'Adam' or self.optimizer =='SGD', "\tUnsupported optimizer: \n\t\tuse 'Adam' or 'SGD' instead."
        if self.optimizer == 'Adam':
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self._beta_1, beta_2=self._beta_2, epsilon=self._epsilon)
        elif self.optimizer == 'SGD':
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self._momentum)

    def _count_loss(self, y_true, y_pred):
        # y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def _count_accuracy(self, y_true, y_pred):
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def _count_f1(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        depth = y_pred._shape_as_list()[2]
        # y_pred = tf.math.argmax(y_pred, axis=-1,  output_type=tf.dtypes.int32)
        y_true = tf.cast(y_true, dtype=tf.dtypes.int32)
        # y_pred = tf.one_hot(y_pred, depth=depth)
        y_true = tf.one_hot(y_true, depth=depth)
        y_true = tf.cast(y_true, dtype=tf.dtypes.float32)
        # y_pred = tf.math.l2_normalize(y_pred, axis=1)
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_score = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
        return f1_score

    def fit(self, data: pd.DataFrame = None, path: str = None):
        """Training function
        """
        assert data is not None or path is not None
        if data is not None:
            self._data_controller = Corpus(lang=self.lang, corpus=data, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)
        else:
            data = pd.read_csv(f'{path}/tokinzed_dataset.csv', sep=';', compression='gzip')
            convertation = lambda d: [[int(z) for z in y] for y in [re.sub("\n", "", re.sub(" +", ",", re.sub(r'[^0-9\s]', '', x))).split(sep=',') for x in d]]
            data['tokenized_answers'] = convertation(data['tokenized_answers'])
            data['tokenized_questions'] = convertation(data['tokenized_questions'])

            with open(f'{path}/metadata.info', 'r', encoding='utf-8') as file:
                temp_d = json.load(file)
            self.max_length = temp_d['max_sent_len']
            self.batch_size = temp_d['batch_size']
            self.buffer_size = temp_d['buffer_size']
            self._data_controller = Corpus(lang=self.lang, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)
            self._data_controller.load(path=path)
        print(f"Data loaded with hyperparams: {self._data_controller}.")
        # print(f'{self._data_controller.dataset.cardinality().numpy()}')
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Embedding(self._data_controller._vocab_size, 256, input_length=len(data['tokenized_questions']), input_shape=(len(data['tokenized_questions'][0]),)))
        self._model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
        self._model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu')))
        self._model.add(tf.keras.layers.Dropout(0.5))
        self._model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self._data_controller._vocab_size, activation='softmax')))
        inputs = np.asarray([np.asarray(xi) for xi in data['tokenized_questions']])
        outputs = np.asarray([np.asarray(xi) for xi in data['tokenized_answers']])
        # self._model.compile(optimizer=self._optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=[self._count_accuracy, self._count_f1])
        self._model.compile(optimizer=self._optimizer, loss=self._count_loss, metrics=[self._count_accuracy, self._count_f1])
        print(f"{self._model} compiled successfully.")
        self._model.fit(inputs, outputs, epochs=self.num_epoch)

    def pred(self, inp:str = 'hi how are you ?'):
        """Prediction for a sentence
        """
        inp:list = self._data_controller._start_token + self._data_controller._tokenizer.encode(inp) + self._data_controller._end_token
        while len(inp) < self.max_length:
            inp.append(0)
        inp = np.asarray([np.asarray(inp)])
        prediction = self._model.predict(inp)
        print(tf.math.argmax(prediction, axis=-1))
        print(self._data_controller._tokenizer.decode([i for i in tf.math.argmax(prediction, axis=-1)[0] if i < self._data_controller._vocab_size-2]))

    def load(self, path:str):
        '''
        Function to initialize model with loading
        '''
        with open(f'{path}/metadata.info', 'r', encoding='utf-8') as file:
            temp_d = json.load(file)
        self.num_epoch = temp_d['num_epoch']
        self.lang = temp_d['lang']
        self.max_length = temp_d['max_length']
        self.batch_size = temp_d['batch_size']
        self.buffer_size = temp_d['buffer_size']
        self._model = tf.keras.models.load_model(path)

    def save_to_folder(self, path:str = None):
        """Saves model to folder
        """
        self._model.save(f"{path}/", overwrite=True, include_optimizer=False)
