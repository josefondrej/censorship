import numpy as np
import pickle as pkl
from typing import Dict, List, Tuple
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense

from language_models.in_memory_model import InMemoryModel
import utils


class BasicLSTM(InMemoryModel):
    def __init__(self, corpus_file_path: str = "./data/data.txt", seq_length: int = 30):
        super().__init__(corpus_file_path)
        self._seq_length = seq_length

    def predict(self, text: str) -> Dict[str, float]:
        corpus = self._clean_corpus(text)
        tokenized_corpus = self._tokenize_corpus(corpus)
        tokens = self._tokenizer.texts_to_sequences(tokenized_corpus)
        if len(tokens) < self._seq_length:
            raise ValueError(f"Seed text is too short. Has to be at least {self._seq_length} words.")

        tokens = np.array(tokens)[-self._seq_length:].reshape(-1, self._seq_length)
        probas = self._model.predict(tokens)[0]  # peel off batch dim
        prediction = {self._index_to_word.get(i): probas[i] for i in range(self.vocab_len)}
        return prediction

    def train(self, batch_size: int = 256, epochs: int = 50):
        self._init_tokenizer()
        self._contexts, self._target_words_one_hot = self._prepare_training_data(self.tokenized_corpus)
        self._model = self._build_model()
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model.fit(self._contexts, self._target_words_one_hot, batch_size=batch_size, epochs=epochs)

    def save_model(self, name: str = "test", model_directory: str = utils.MODEL_DIRECTORY):
        model_file_path, tokenizer_file_path = self._get_paths(name, model_directory)
        self._model.save(model_file_path)
        pkl.dump(self._tokenizer, open(tokenizer_file_path, "wb"))

    def load_model(self, name: str = "test", model_directory: str = utils.MODEL_DIRECTORY):
        model_file_path, tokenizer_file_path = self._get_paths(name, model_directory)
        self._model = load_model(model_file_path)
        self._tokenizer = pkl.load(open(tokenizer_file_path, "rb"))

    def _get_paths(self, name: str = "test", model_directory: str = utils.MODEL_DIRECTORY):
        model_file_path = model_directory + name + "_model.h5"
        tokenizer_file_path = model_directory + name + "_tokenizer.pkl"
        return model_file_path, tokenizer_file_path

    def _init_tokenizer(self):
        self._tokenizer = Tokenizer()
        self._tokenizer.fit_on_texts(self._tokenized_corpus)
        self._index_to_word = dict(map(reversed, self._tokenizer.word_index.items()))

    def _prepare_training_data(self, tokenized_corpus: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tokenized_corpus = self._tokenizer.texts_to_sequences(tokenized_corpus)
        tokenized_corpus = utils.flatten(tokenized_corpus)
        contexts = []
        target_words_one_hot = []
        for i in range(self.vocab_len // (self._seq_length + 1)):
            start, end = i * (self._seq_length + 1), (i + 1) * (self._seq_length + 1) - 1
            context, target_word = tokenized_corpus[start:end], tokenized_corpus[end]
            target_word_one_hot = to_categorical(target_word, num_classes=self.vocab_len)
            contexts.append(context)
            target_words_one_hot.append(target_word_one_hot)

        return np.array(contexts), np.array(target_words_one_hot)

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(Embedding(self.vocab_len, 64, input_length=self._seq_length))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.vocab_len, activation='softmax'))
        return model


if __name__ == "__main__":
    seed_text = """The girls came just in time; they held him fast and tried to free his
                    beard from the line, but all in vain, beard and line were entangled fast
                    together. There was nothing to do but to bring out the scissors and cut
                    the beard, whereby a small part of it was lost. When the dwarf saw that
                    he screamed out: ‘Is that civil, you toadstool, to disfigure a man’s
                    face? Was it not enough to clip off the end of my beard? Now you have
                    """

    basic_lstm = BasicLSTM("./data/grimm.txt")

    print(f"Vocabulary length: {basic_lstm.vocab_len}")
    print(f"Vocabulary example: {list(basic_lstm.vocabulary)[0:500]}")

    basic_lstm.train(epochs=100)
    prediction = basic_lstm.predict(seed_text)

    print(f"Total proba (should be close to 1.0): {sum(prediction.values())}")
    print("First few proba predictions: ", {k: v for t, (k, v) in enumerate(prediction.items()) if t < 10})
