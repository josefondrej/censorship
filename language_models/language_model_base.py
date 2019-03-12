from typing import Dict, Set


class LanguageModelBase(object):
    def __init__(self, corpus_file_path: str = "./data/data.txt"):
        self._corpus_file_path = corpus_file_path
        self._vocabulary: Set = None

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def vocab_len(self):
        return len(self._vocabulary)

    def train(self):
        raise NotImplementedError("Has to be overriden.")

    def predict(self, text: str) -> Dict[str, float]:
        # import numpy as np
        # from utils import softmax
        # n = len(self.vocabulary)
        # probas = softmax(np.random.randn(n))
        # predictions = dict(zip(self.vocabulary, probas))
        # return predictions
        raise NotImplementedError("Has to be overriden.")
