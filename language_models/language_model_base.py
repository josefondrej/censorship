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
        """ Returns {word: log-proba of word given text} """
        raise NotImplementedError("Has to be overriden.")
