"""Main transformer NN module

Note: most of this code is based on [Tensorflow transformer guide](https://www.tensorflow.org/text/tutorials/transformer)

The module contains:
    1. MHA
    2. Positional encoding
    3. Step-based shedule
    4. Transformer model
"""
from dataclasses import dataclass, field
import tensorflow as tf
import pandas as pd
from preprocessor import Corpus


assert tf.__version__.startswith('2')
tf.random.set_seed(42)

# pylint: disable=line-too-long
# pylint: disable=protected-access
class MultiHeadAttention(tf.keras.layers.Layer):
    """Realization of [Multi-Head Attention algorithm](https://arxiv.org/abs/1706.03762v5)

    This class inherits from default [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) and overrites it.

    _Exampe Usage_

    ```python
    d_model = 256
    num_heads = 8

    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention = MultiHeadAttention(d_model, num_heads, name="attention")(inputs={'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
    ```

    Args:
        num_heads: 
        d_model: size of computed space (d-dimensional)
        name: (optional str) the `tf.keras.Layer`'s name
    """
    def __init__(self, d_model:int, num_heads:int, name: str="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size:int):
        """Reshape function for preparing SCALDE
        """
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        """SCALDE function. Calculates dot product of independent q-,k-,v-dimensional vectors.
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        if mask is not None:
            logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def call(self, inputs):
        """`tf.keras.layers.Layer.call()` function overriting for use this MHA.
        """
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        return outputs

class StepBasedShedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom step-based optimized shedule for learning rate. Recommended by Ashish Vaswani.

    Note: This class inherits from default [`tf.keras.optimizers.schedules.LearningRateSchedule`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule) and overrites it.

    _Exampe Usage_

    ```python
    d_model = 256

    learning_rate_shedule = StepBasedShedule(d_model)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate_shedule)
    ```    
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(StepBasedShedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1, arg2 = tf.math.rsqrt(step),  step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {'d_model': self.d_model,'warmup_steps': self.warmup_steps,}
        return config

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

@dataclass(frozen=False, kw_only=True, slots=True)
class Transformer():
    """Main transformer class
    """
    num_layers: int = field(default=3, init=True, repr=True)
    num_heads: int = field(default=8, init=True, repr=True)
    num_epoch: int = field(default=1, init=True, repr=True)
    units: int = field(default=512, init=True, repr=True)
    treshold: float = field(default=0.1, init=True, repr=True)
    d_model: int = field(default=256, init=False, repr=True)
    lang: str = field(default="en-us", init=True, repr=True)
    max_length: int = field(default=40, init=True, repr=True)
    batch_size: int = field(default=64, init=True, repr=True)
    buffer_size: int = field(default=20000, init=True, repr=True)
    optimizer: str = field(default='Adam', init=True, repr=True)
    learning_rate_shedule: tf.keras.optimizers.schedules.LearningRateSchedule = field(default=None, init=True, repr=True)
    _optimizer: tf.keras.optimizers.Adam = field(default=None, init=False, repr=False)
    _beta_1: float = field(default=0.9, init=True, repr=False)
    _beta_2: float = field(default=0.98, init=True, repr=False)
    _epsilon: float = field(default=1e-9, init=True, repr=False)
    _momentum: float = field(default=0, init=True, repr=False)
    _data_controller: Corpus = field(default_factory=Corpus, init=False, repr=False)
    _model: tf.keras.Model = field(default_factory=tf.keras.Model, init=False, repr=False)

    def __post_init__(self):
        tf.keras.backend.clear_session()
        self.learning_rate_shedule = StepBasedShedule(self.d_model)
        assert self.optimizer == 'Adam' or self.optimizer =='SGD', "\tUnsupported optimizer: \n\t\tuse 'Adam' or 'SGD' instead."
        if self.optimizer == 'Adam':
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_shedule, beta_1=self._beta_1, beta_2=self._beta_2, epsilon=self._epsilon)
        elif self.optimizer == 'SGD':
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate_shedule, momentum=self._momentum)

    def _count_accuracy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def _count_loss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def _create_decoder_layer(self, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
        attention1 = MultiHeadAttention(self.d_model, self.num_heads, name="attention_1")(inputs={'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
        attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
        attention2 = MultiHeadAttention(self.d_model, self.num_heads, name="attention_2")(inputs={'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
        attention2 = tf.keras.layers.Dropout(rate=self.treshold)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)
        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.treshold)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

    def decoder(self):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
        embeddings = tf.keras.layers.Embedding(self._data_controller._vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self._data_controller._vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=self.treshold)(embeddings)
        for i in range(self.num_layers):
            outputs = self._create_decoder_layer(name=f'decoder_layer_{i}')(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name='decoder')

    def _create_encoder_layer(self, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        attention = MultiHeadAttention(self.d_model, self.num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
        attention = tf.keras.layers.Dropout(rate=self.treshold)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.treshold)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def encoder(self):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        embeddings = tf.keras.layers.Embedding(self._data_controller._vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self._data_controller._vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=self.treshold)(embeddings)
        for i in range(self.num_layers):
            outputs = self._create_encoder_layer(name=f"encoder_layer_{i}",)([outputs, padding_mask])

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name="encoder")

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def fit(self, data: pd.DataFrame):
        self._data_controller = Corpus(lang=self.lang, corpus=data, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)

        print(f"Data loaded with hyperparams: {self._data_controller}.")

        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(self.create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])
        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
        outputs = tf.keras.layers.Dense(units=self._data_controller._vocab_size, name="outputs")(dec_outputs)

        self._model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name="transformer")
        self._model.compile(optimizer=self._optimizer, loss=self._count_loss, metrics=[self._count_accuracy])

        print(f"{self._model} compiled successfully.")
        self._model.fit(self._data_controller.dataset, epochs=self.num_epoch)

    def fit_test(self, data):

        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(self.create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])
        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
        outputs = tf.keras.layers.Dense(units=self._data_controller._vocab_size, name="outputs")(dec_outputs)

        self._model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name="transformer")
        self._model.compile(optimizer=self._optimizer, loss=self._count_loss, metrics=[self._count_accuracy])

        print(f"{self._model} compiled successfully.")
        self._model.fit(data, epochs=self.num_epoch)