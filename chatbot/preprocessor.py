"""Module with text controller class.

Warning: Corpus class is frozen @dataclass. Once you create an instance,
you couldn't change anything from outside the class.
"""
import re
from dataclasses import dataclass, field
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

__all__ = ["Corpus"]
# pylint: disable=line-too-long

@dataclass(frozen=False, kw_only=True, slots=True)
class Corpus():
    """Is using for locate datasets and text filtering functions.

    Warning: a class instance can take up a lot of memory space when using a large dictionary.

    For example,
    >>> corpus = Corpus(lang='ru', max_length=20, corpus=corpus)
    >>> print(Corpus)
    >>> is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))89-

    Args:
        lang: (optional string [default 'en-us']) set the lang; list of available languages you can find in guide.
        max_length: (optional int) maximum length of the sentence to train/predict
        batch_size: (optional int) size of one batch
        buffer_size: (optional int) size of the buffer
        corpus: (optional pd.Dataframe) dataframe with questions-answers columns; is not nessesary when loading from saved model;

    Attention: Note that this class is highly not recommended to be used from scratch.
    We recommend to use build in functions in Model class.

    Consist:
        start_token: id_key for the start of the sentence
        end_token: id_key for the end of the sentence
        vocab_size: size of constructed vocabulary based on SubwordTextEncoder
        dataset: (tf.data.Dataset) final dataset to train transformer
    """
    lang: str = field(default="en", init=True, repr=True)
    max_length: int = field(default=40, init=True, repr=True)
    batch_size: int = field(default=64, init=True, repr=True)
    buffer_size: int = field(default=20000, init=True, repr=True)
    _start_token: int = field(default=0, init=False, repr=True)
    _end_token: int = field(default=0, init=False, repr=True)
    _vocab_size: int = field(default=0, init=False, repr=True)
    corpus: pd.DataFrame = field(default=pd.DataFrame, init=True, repr=False)
    _questions: list = field(default_factory=list, init=False, repr=False)
    _answers: list = field(default_factory=list, init=False, repr=False)
    _tokenizer: tfds.deprecated.text.SubwordTextEncoder = field(default_factory=list, init=False, repr=False)
    dataset: tf.data.Dataset = field(default_factory=list, init=False, repr=False)
    _sent_len: list = field(default_factory=list, init=False, repr=False)
    _fre_full: list = field(default_factory=list, init=False, repr=False)
    fre: float = field(default=0.0, init=False, repr=True)
    av_sent_len: float = field(default=0.0, init=False, repr=True)

    def __post_init__(self) -> None:
        if not self.corpus.empty:
            print(f'{tf.data.Dataset}: Dictionary generation {self}. \n\tPlease be patient. Depending on the size of the dataset and \n\tthe number of unique words, this can take a while (up to 5 minutes).')
            self.corpus.dropna(inplace=True)
            self.create()

    def create(self):
        """Function to create new dataset
        """
        # pylint: disable=unsubscriptable-object
        self._questions = [self.preprocess_sentence(sent) for sent in self.corpus.iloc[:, 0].to_list()]
        self._answers = [self.preprocess_sentence(sent) for sent in self.corpus.iloc[:, 1].to_list()]
        self._tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(self._questions + self._answers, target_vocab_size=2**16)
        self._start_token, self._end_token = [self._tokenizer.vocab_size], [self._tokenizer.vocab_size + 1]
        self._vocab_size = self._tokenizer.vocab_size + 2
        self._tokenize_and_filter()
        self._create_dataset()

    def _count_fre(self, sentence:str):
        sentence = re.sub(", ", '', sentence)
        sentence = re.sub("[!?.:]", '.', sentence)
        while sentence.count('. .') >0:
            sentence = sentence.replace('. .', '.')
        words_c = len([w for w in sentence.split(' ') if w !='.' and w])
        if words_c == 0:
            return None
        sentence_c  = len([w for w in sentence.split(' ') if w =='.'])+1
        syllable_c = 0
        if self.lang == 'en':
            for word in sentence:
                if word in ('a', 'e', 'i', 'o', 'u'):
                    syllable_c+=1
            self._fre_full.append(206.835 - 1.015 * (words_c/sentence_c)-84.6*(syllable_c/words_c))
        elif self.lang == 'ru':
            for word in sentence:
                if word in ('??', '??', '??', '??', '??', '??', '??', '??', '??'):
                    syllable_c+=1
            self._fre_full.append(206.835 - 1.3 * (words_c/sentence_c)-60.1*(syllable_c/words_c))

    def preprocess_sentence(self, sentence: str) -> str:
        """Function for single sentence preprocessing.

        Works with english and russian characters now. To implement other lang just add
        ```python
        elif self.lang == YOUR_LANGUAGE:
            sentence = re.sub(r"[^YOUR_LANGUAGE_CHARACTERS?.!,: ]+", "", sentence)
        ```
        and comment `assert` and `self._count_fre()` rows

        Typical usage:
        >>> corpus = Corpus(lang='ru', max_length=20, corpus=corpus)
        >>> sent = 'Hi, I'm here.!'
        >>> corpus.preprocess_sentence(sent)
        >>> <<< 'hi , i m here !'

        Args:
            sentence: string with you sentence(s)

        Returns:
            sentence: filtered sentence

        Hidden:
            fre: updates fre for supported langs
        """
        assert self.lang == 'en' or self.lang =='ru', "\tUnsupported language: \n\t\tuse 'ru' or 'en' instead."
        sentence = sentence.lower().strip()
        sentence = sentence.replace('\n', '')
        if self.lang == 'ru':
            sentence = re.sub(r"[^??-????-???.!,: ]+", "", sentence)
        elif self.lang == 'en':
            sentence = re.sub(r"[^a-zA-Z?.!,: ]+", "", sentence)
        sentence = re.sub(r' *([.,!?:]) *', r' \1 ', sentence)
        sentence = [value for value in sentence.split(' ') if value]
        sentence = ' '.join(sentence)
        while sentence.count('? ?') >0:
            sentence = sentence.replace('? ?', '?')
        while sentence.count('. .') >0:
            sentence = sentence.replace('. .', '.')
        while sentence.count('! !') >0:
            sentence = sentence.replace('! !', '!')
        while sentence.count(', ,') >0:
            sentence = sentence.replace(', ,', ',')
        while sentence.count(': :') >0:
            sentence = sentence.replace(': :', ':')
        sentence = sentence.replace('! ?', '? !')

        sentence = sentence.replace(': .', ':')
        sentence = sentence.replace('? :', ':')
        sentence = sentence.replace(', :', ':')
        sentence = sentence.replace('! :', ':')
        sentence = sentence.replace(': !', ':')
        sentence = sentence.replace(': ?', ':')
        sentence = sentence.replace('. :', ':')
        sentence = sentence.replace(': ,', ':')

        sentence = sentence.replace('! .', '!')
        sentence = sentence.replace('? .', '?')
        sentence = sentence.replace(', .', ',')
        sentence = sentence.replace('. !', '!')
        sentence = sentence.replace('. ?', '?')
        sentence = sentence.replace('. ,', ',')
        sentence = sentence.replace('! ,', '!')
        sentence = sentence.replace('? ,', '?')
        sentence = sentence.replace(', !', '!')
        sentence = sentence.replace(', ?', '?')
        sentence = sentence.strip()
        self._count_fre(sentence)
        return sentence

    def _tokenize_and_filter(self):
        tokenized_inputs, tokenized_outputs = [], []
        for (sentence1, sentence2) in zip(self._questions, self._answers):
            sentence1 = self._start_token + self._tokenizer.encode(sentence1) + self._end_token
            sentence2 = self._start_token + self._tokenizer.encode(sentence2) + self._end_token
            if len(sentence1) <= self.max_length and len(sentence2) <= self.max_length:
                self._sent_len.append(len(sentence1))
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        self._questions = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.max_length, padding='post')
        self._answers = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.max_length, padding='post')

    def _create_dataset(self):
        # pylint: disable=invalid-sequence-index
        self.dataset = tf.data.Dataset.from_tensor_slices((
            {'inputs': self._questions,'dec_inputs': self._answers[:, :-1]},
            {'outputs': self._answers[:, 1:]},))
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(self.buffer_size)
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def save(self, path: str = None):
        """Function to save current created instance
        """
        self._tokenizer.save_to_file(f'{path}/tokenizer.tf')
        self.dataset.save(path=f"{path}/dataset", compression='GZIP')

    def _load(self, path:str, batch_size, lang, max_length, buffer_size):
        self._tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f'{path}/tokenizer.tf')
        self._start_token, self._end_token = [self._tokenizer.vocab_size], [self._tokenizer.vocab_size + 1]
        self._vocab_size = self._tokenizer.vocab_size + 2
        self.batch_size = batch_size
        self.lang = lang
        self.max_length = max_length
        self.buffer_size = buffer_size


    def load(self, path: str = None):
        """Function to load presaved instance
        """
        self._tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f'{path}/tokenizer.tf')
        self.dataset = tf.data.Dataset.load(path=f"{path}/tfds", compression='GZIP')
        self._start_token, self._end_token = [self._tokenizer.vocab_size], [self._tokenizer.vocab_size + 1]
        self._vocab_size = self._tokenizer.vocab_size + 2
