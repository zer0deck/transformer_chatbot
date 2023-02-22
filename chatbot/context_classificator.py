"""
Data tagger model. Reference on default Tensorflow classificator:
https://www.tensorflow.org/tutorials/keras/text_classification?hl=ru
"""

import re
import string
from dataclasses import dataclass, field
import tensorflow as tf

CLASSES = {
    0: 'atheism',
    1: 'autos',
    2: 'baseball',
    3: 'christian',
    4: 'crypt',
    5: 'electronics',
    6: 'graphics',
    7: 'guns',
    8: 'hardware',
    9: 'hockey',
    10: 'medical',
    11: 'mideast',
    12: 'motorcycles',
    13: 'os',
    14: 'politics',
    15: 'religion',
    16: 'sale',
    17: 'space',
    18: 'windows'
}

@tf.keras.utils.register_keras_serializable()
def preprocessor(input_data:tf.Tensor):
    """Preprocess sentences as tensor to classificate"""
    lowercase = tf.strings.lower(input_data)
    sentence = tf.strings.regex_replace(lowercase, r' *([.,!?:]) *', r' \1 ')
    stripped_html = tf.strings.regex_replace(sentence, '^From.+', ' ')
    stripped_html2 = tf.strings.regex_replace(stripped_html, '^Subject.+', ' ')
    rez = tf.strings.regex_replace(stripped_html2,
        f'[{re.escape(string.punctuation)}]', '')
    return tf.strings.regex_replace(rez, " +", " ")

@dataclass(frozen=False, kw_only=True, slots=True)
class ContextClassificator():
    batch_size: int = field(default=32, init=True, repr=True)
    seed: int = field(default=42, init=True, repr=True)
    max_features: int = field(default=10000, init=True, repr=True)
    sequence_length: int = field(default=250, init=True, repr=True)
    autotune: None = field(default=tf.data.AUTOTUNE, init=False, repr=False)
    embedding_dim: int = field(default=16, init=True, repr=True)
    model: tf.keras.Model = field(default_factory=tf.keras.Model, init=False, repr=False)
    history: tf.keras.callbacks.History = field(
        default_factory=tf.keras.callbacks.History, init=False, repr=False)
    vectorize_layer: None = field(default=None, init=False, repr=False)
    _is_loaded: bool = field(default=False, init=False, repr=True)

    def train(self, path: str, epochs: int = 20):
        """Train classification model"""
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            path,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='training',
            seed=self.seed)
        raw_val_ds = tf.keras.utils.text_dataset_from_directory(
            path,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation',
            seed=self.seed)

        self.vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=preprocessor,
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length)

        train_text = raw_train_ds.map(lambda x, y: x)
        self.vectorize_layer.adapt(train_text)

        tmp_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.max_features + 1, self.embedding_dim),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(19)])

        tmp_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer='adam',
                    metrics=['accuracy'])

        self.model = tf.keras.Sequential([
            self.vectorize_layer,
            tmp_model,
            tf.keras.layers.Activation('sigmoid')
            ])

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer='adam',
                    metrics=['accuracy'])

        self.history = self.model.fit(
            raw_train_ds,
            validation_data=raw_val_ds,
            epochs=epochs)

        loss, accuracy = self.model.evaluate(raw_val_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        return self

    def save_to_folder(self, path:str = None):
        """Saves model to folder
        """
        self.model.save(f"{path}/classificator/", save_format='tf')

    def load(self, path:str = None):
        """Loads model from folder"""
        # from_disk = pickle.load(open(f"{path}/classificator/tvl.pkl", "rb"))
        # self.vectorize_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
        # self.vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        # self.vectorize_layer.set_weights(from_disk['weights'])
        self.model = tf.keras.models.load_model(f"{path}/classificator")
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer='adam',
                    metrics=['accuracy'])
        self._is_loaded = True
        return self

    def predict(self, sentence:str) -> str:
        """
        Predicts text tag

        Returns:
            predicted_class: str
        """
        rez = tf.argmax(self.model.predict(tf.constant([sentence]), verbose=0), axis=1)
        return CLASSES[rez.numpy()[0]]
