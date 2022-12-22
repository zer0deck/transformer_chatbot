# pylint: disable=[line-too-long, protected-access, arguments-differ]
"""Sequential model file
"""
import json
from dataclasses import dataclass, field
import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
from chatbot.preprocessor import Corpus

@dataclass(frozen=False, kw_only=True, slots=True)
class Sequent():
    """Default multilayer perceptron model
    """
    num_epoch: int = field(default=1, init=True, repr=True)
    treshold: float = field(default=0.1, init=True, repr=True)
    d_model: int = field(default=512, init=False, repr=True)
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

    def fit(self, data: pd.DataFrame = None, path: str = None):

        assert data is not None or path is not None
        if data is not None:
            self._data_controller = Corpus(lang=self.lang, corpus=data, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)
        else:
            data = pd.read_csv(f'{path}/tokinzed_dataset.csv', sep=';', compression='gzip')
            print(data['tokenized_answers'][0])
            with open(f'{path}/metadata.info', 'r', encoding='utf-8') as file:
                temp_d = json.load(file)
            self.max_length = temp_d['max_sent_len']
            self.batch_size = temp_d['batch_size']
            self.buffer_size = temp_d['buffer_size']
            self._data_controller = Corpus(lang=self.lang, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)
            self._data_controller.load(path=path)
        print(f"Data loaded with hyperparams: {self._data_controller}.")
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(self.d_model, input_shape=(15,), activation='relu'))
        self._model.add(tf.keras.layers.Dropout(self.treshold))
        temp_a = self.d_model
        while temp_a > 64:
            temp_a = temp_a // 2
            self._model.add(tf.keras.layers.Dense(temp_a, activation='relu'))
            self._model.add(tf.keras.layers.Dropout(self.treshold))
        self._model.add(tf.keras.layers.Dense(15, activation='softmax'))

        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self._model.compile(optimizer=self._optimizer, loss='mse', metrics=['accuracy', tfr.keras.metrics.MRRMetric()])
        print(f"{self._model} compiled successfully.")
        self._model.fit(self._data_controller.dataset, epochs=self.num_epoch)
    
    def predict(self, input: list[int]):
        self._model.predict()