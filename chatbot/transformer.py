# pylint: disable=[line-too-long, protected-access, arguments-differ, no-value-for-parameter]
"""Main transformer NN module

Note: most of this code is based on [Tensorflow transformer guide](https://www.tensorflow.org/text/tutorials/transformer)

The module contains:
    1. MHA
    2. Positional encoding
    3. Step-based shedule
    4. Transformer model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import json
from dataclasses import dataclass, field
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from chatbot.preprocessor import Corpus, preprocess_sentence


assert tf.__version__.startswith('2')
tf.random.set_seed(42)

@tf.keras.utils.register_keras_serializable()
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
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {'d_model': self.d_model,'warmup_steps': self.warmup_steps,}
        return config

# pylint: disable=[missing-class-docstring, missing-function-docstring, unexpected-keyword-arg]
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
# pylint: enable=[missing-class-docstring, missing-function-docstring, unexpected-keyword-arg]

@dataclass(frozen=False, kw_only=True, slots=True)
class Transformer():
    """Main transformer class
    """
    num_layers: int = field(default=4, init=True, repr=True)
    num_heads: int = field(default=8, init=True, repr=True)
    num_epoch: int = field(default=1, init=True, repr=True)
    units: int = field(default=512, init=True, repr=True)
    treshold: float = field(default=0.1, init=True, repr=True)
    d_model: int = field(default=256, init=False, repr=True)
    lang: str = field(default="en", init=True, repr=True)
    max_length: int = field(default=30, init=True, repr=True)
    batch_size: int = field(default=32, init=True, repr=True)
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
    _context_classificator: tf.keras.Model = field(default=None, init=True, repr=False)
    _emotion_classificator: tf.keras.Model = field(default=None, init=True, repr=False)
    _classificators_loaded: bool = field(default=False, init=False, repr=True)
    history: tf.keras.callbacks.History = field(default_factory=tf.keras.callbacks.History, init=False, repr=False)
    mrr: tf.Variable = field(default=None, init=False, repr=False)

    def __post_init__(self):
        tf.keras.backend.clear_session()
        self.learning_rate_shedule = StepBasedShedule(self.d_model)
        assert self.optimizer == 'Adam' or self.optimizer =='SGD', "\tUnsupported optimizer: \n\t\tuse 'Adam' or 'SGD' instead."
        if self.optimizer == 'Adam':
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_shedule, beta_1=self._beta_1, beta_2=self._beta_2, epsilon=self._epsilon)
        elif self.optimizer == 'SGD':
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate_shedule, momentum=self._momentum)
        if self._context_classificator or self._emotion_classificator:
            self._classificators_loaded = True

    def _count_accuracy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def _count_f1(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true = tf.reshape(y_true, shape=(-1, self.max_length - 1))
        depth = y_pred._shape_as_list()[2]
        y_pred = tf.math.argmax(y_pred, axis=-1,  output_type=tf.dtypes.int32)
        y_true = tf.cast(y_true, dtype=tf.dtypes.int32)
        y_pred = tf.one_hot(y_pred, depth=depth)
        y_true = tf.one_hot(y_true, depth=depth)
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_score = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
        return f1_score

    def _count_mrr(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true = tf.reshape(y_true, shape=(-1, 1))
        y_pred = tf.reshape(y_pred, shape=(-1, self._data_controller._vocab_size))
        y_pred = tf.math.top_k(y_pred, k=self._data_controller._vocab_size).indices
        y_true = tf.cast(y_true, dtype=tf.dtypes.int32)
        where_tensor = tf.equal(y_true,y_pred)
        where_tensor = tf.where(where_tensor)[:, 1]
        where_tensor = tf.cast(tf.add(where_tensor, 1), dtype=tf.dtypes.float64)
        where_tensor = tf.divide(tf.constant(np.ones(where_tensor._shape_as_list()[0])), where_tensor)
        return tf.math.reduce_mean(where_tensor)

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
        """Decoder
        """
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
        """Encoder
        """
        input1 = tf.keras.Input(shape=(None,), name="input1")
        input2 = tf.keras.Input(shape=(None,), name="input2")
        input3 = tf.keras.Input(shape=(None,), name="input3")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embedding1 = tf.keras.layers.Embedding(self._data_controller._vocab_size, self.d_model)(input1)
        embedding1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        embedding2 = tf.keras.layers.Embedding(self._data_controller._vocab_size, self.d_model)(input2)
        embedding2 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        embedding3 = tf.keras.layers.Embedding(self._data_controller._vocab_size, self.d_model)(input3)
        embedding3 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        embeddings = PositionalEncoding(self._data_controller._vocab_size, self.d_model)(embedding1 + embedding2 + embedding3)

        outputs = tf.keras.layers.Dropout(rate=self.treshold)(embeddings)
        for i in range(self.num_layers):
            outputs = self._create_encoder_layer(name=f"encoder_layer_{i}",)([outputs, padding_mask])

        return tf.keras.Model(inputs=[input1, input2, input3, padding_mask], outputs=outputs, name="encoder")

# pylint: disable = [invalid-name]
    def create_padding_mask(self, x):
        """Makes Pmask for traing
        """
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        """Makes LAmask for traing
        """
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)
# pylint: enable = [invalid-name]

    def _train_emo(self, num_classes:int = 8, num_epochs: int = 5):
        self._emotion_classificator = tf.keras.Sequential()
        self._emotion_classificator.add(tf.keras.layers.Embedding(self._data_controller._vocab_size, 200))
        self._emotion_classificator.add(tf.keras.layers.LSTM(128))
        self._emotion_classificator.add(tf.keras.layers.Dropout(0.2))
        self._emotion_classificator.add(tf.keras.layers.Dense(32, activation = 'relu'))
        self._emotion_classificator.add(tf.keras.layers.Dropout(0.2))
        self._emotion_classificator.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
        self._emotion_classificator.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['AUC', 'Recall', 'Precision', 'accuracy'])
        self._emotion_classificator.fit(self._data_controller.emo_df, epochs=num_epochs)
        self._emotion_classificator.save('trained_models/emotion_detector', overwrite=True)

    def _train_context(self, num_classes:int = 4, num_epochs: int = 5):
        self._context_classificator = tf.keras.Sequential()
        self._context_classificator.add(tf.keras.layers.Embedding(self._data_controller._vocab_size, 200))
        self._context_classificator.add(tf.keras.layers.LSTM(128))
        self._context_classificator.add(tf.keras.layers.Dropout(0.2))
        self._context_classificator.add(tf.keras.layers.Dense(32, activation = 'relu'))
        self._context_classificator.add(tf.keras.layers.Dropout(0.2))
        self._context_classificator.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
        self._context_classificator.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['AUC', 'Recall', 'Precision', 'accuracy'])
        self._context_classificator.fit(self._data_controller.cont_df, epochs=num_epochs)
        self._context_classificator.save('trained_models/classificator', overwrite=True)

    def fit_data(self, data: pd.DataFrame):
        """Training function
        """
        self._data_controller = Corpus(lang=self.lang, corpus=data, max_length=self.max_length, batch_size=self.batch_size, buffer_size=self.buffer_size)

    def train_classificators(self, num_ep: int = 5):
        """...
        """
        self._data_controller._create_class_dataset()
        self._train_emo(num_epochs=num_ep)
        del self._emotion_classificator
        del self._data_controller.emo_df
        self._train_context(num_epochs=num_ep)
        del self._context_classificator
        del self._data_controller.cont_df
        with open('trained_models/cont_encoder.json', 'w', encoding='utf8') as file_object:
            json.dump({int(k): int(v) for (k, v) in self._data_controller.cont_encoder.items()}, file_object)
        with open('trained_models/emo_encoder.json', 'w', encoding='utf8') as file_object:
            json.dump({int(k): int(v) for (k, v) in self._data_controller.emo_encoder.items()}, file_object)

    def compile_model(self):
        """Training function
        """
        self._data_controller._create_dataset()
        print(f"Data loaded with hyperparams: {self._data_controller}.")
        tf.keras.backend.clear_session()
        input1 = tf.keras.Input(shape=(None,), name="input1")
        input2 = tf.keras.Input(shape=(None,), name="input2")
        input3 = tf.keras.Input(shape=(None,), name="input3")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
        enc_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(input1)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(self.create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(input1)

        enc_outputs = self.encoder()(inputs=[input1, input2, input3, enc_padding_mask])
        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
        outputs = tf.keras.layers.Dense(units=self._data_controller._vocab_size, name="outputs")(dec_outputs)

        self._model = tf.keras.Model(inputs=[input1, input2, input3, dec_inputs], outputs=outputs, name="transformer")
        self._model.compile(optimizer=self._optimizer, loss=self._count_loss, metrics=[self._count_accuracy, self._count_f1, self._count_mrr])
        # self._model.compile(optimizer=self._optimizer, loss=self._count_loss, metrics=[self._count_accuracy, self._count_f1])


        # CHECKPOINT MANAGER:

        checkpoint = tf.keras.callbacks.BackupAndRestore(
            'trained_models/transformer',
            save_freq='epoch',
            delete_checkpoint=False
            )
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='_count_accuracy',
            min_delta=0.002,
            mode="max",
            patience=5,
            verbose=1
            )


        print(f"{self._model} compiled successfully.")
        self.history = self._model.fit(
            self._data_controller.dataset,
            epochs=self.num_epoch,
            verbose=2,
            callbacks=[checkpoint, stop_early]
            )
        return self.history.history

# pylint: disable = [missing-function-docstring, unexpected-keyword-arg, unsubscriptable-object]
    def evaluate(self, sentence: str, k:int=0, tag: str = None, emote: str = None, classificator_path:str = 'trained_models/classificator', emotion_detector_path: str = 'trained_models/emotion_detector', flag: bool = False):

        sentence = tf.expand_dims(self._data_controller._start_token + self._data_controller._tokenizer.encode(sentence) + self._data_controller._end_token, axis=0)
        if flag:
            if not self._classificators_loaded:
                self._context_classificator = tf.keras.models.load_model(classificator_path)
                self._context_classificator.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['AUC', 'Recall', 'Precision', 'accuracy'])
                self._emotion_classificator = tf.keras.models.load_model(emotion_detector_path)
                self._emotion_classificator.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['AUC', 'Recall', 'Precision', 'accuracy'])
            tag = tf.expand_dims(
                int(list(self._data_controller.cont_encoder.keys())[
                    list(self._data_controller.cont_encoder.values())
                    .index(tf.argmax(
                            self._context_classificator.predict(
                                tf.constant(sentence),
                                verbose=0
                            ),
                            axis=1
                        ).numpy()[0]
                    )
                ]),
                axis=0
            )
            emote = tf.expand_dims(
                int(list(self._data_controller.emo_encoder.keys())[
                    list(self._data_controller.emo_encoder.values())
                    .index(tf.argmax(
                            self._context_classificator.predict(
                                tf.constant(sentence),
                                verbose=0
                            ),
                            axis=1
                        ).numpy()[0]
                    )
                ]),
                axis=0
            )
        else:
            tag = tf.expand_dims([tag], axis=0)
            emote = tf.expand_dims(self._data_controller._tokenizer.encode(emote), axis=0)
        output = tf.expand_dims(self._data_controller._start_token, 0)
        for _ in range(self.max_length):
            predictions = self._model(inputs=[sentence, emote, tag, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            _, indices = tf.nn.top_k(predictions[0][0],k+1)
            predicted_id = tf.cast([[indices[k]]], tf.int32)
            # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self._data_controller._end_token[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)
# pylint: enable = [missing-function-docstring, unexpected-keyword-arg, unsubscriptable-object]

    def predict(self, sentence: str, flag=False, tag = None, emote = None):
        """Function to predict sentence
        """
        predictions = []
        for i in range(1):
            predictions.append(self.evaluate(sentence, flag=flag, tag=tag, emote=emote, k=i))

        predicted_sentences = [self._data_controller._tokenizer.decode([i for i in prediction if i < self._data_controller._tokenizer.vocab_size]) for prediction in predictions]

        return predicted_sentences

    def chat(self, data: pd.DataFrame = pd.DataFrame()) -> None | list[str]:
        """
        Chat conversation init
        """
        if data.empty:
            with open('trained_models/emo_encoder.json', 'r', encoding='utf8') as file_object:
                self._data_controller.emo_encoder = json.load(file_object)
            with open('trained_models/cont_encoder.json', 'r', encoding='utf8') as file_object:
                self._data_controller.cont_encoder = json.load(file_object)
            while True:
                text = input('Введите текст:')
                if text == '0':
                    break
                text = preprocess_sentence(text)
                print('You:', text)
                print('Bot:',self.predict(text, flag=True))
            return None
        dec =  {
            "home living": 4708,
            "business": 1349,
            "minorities": 69034,
            "sports": 7243,
            "politics": 3406,
            "wellness": 15507,
            "entertainment": 11989,
            "travel": 2010,
            "culture": 4060
            }
        result = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            tag = dec[row['category']]
            ans = self.predict(row['question'], tag=tag, emote=row['emotion'])
            result.append(ans)
        return result
