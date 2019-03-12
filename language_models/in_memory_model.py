from typing import Dict, List

from language_models.language_model_base import LanguageModelBase
import utils


class InMemoryModel(LanguageModelBase):
    def __init__(self, corpus_file_path: str = "./data/data.txt"):
        super().__init__(corpus_file_path)

        with open(self._corpus_file_path, "r") as corpus_file:
            self._corpus = corpus_file.read()

        self._cleaned_corpus = self._clean_corpus(self._corpus)
        self._tokenized_corpus = self._tokenize_corpus(self._cleaned_corpus)

        del self._corpus, self._cleaned_corpus

        self._init_vocabulary(self._tokenized_corpus)

    @property
    def tokenized_corpus(self):
        return self._tokenized_corpus

    def predict(self, text: str) -> Dict[str, float]:
        raise NotImplementedError("Has to be overriden.")

    def train(self):
        raise NotImplementedError("Has to be overriden.")

    def _clean_corpus(self, corpus: str, translation_dict: Dict[str, str] = utils.TRANSLATE) -> str:
        table = str.maketrans(translation_dict)
        translation = corpus.translate(table)
        translation = translation.lower()
        return translation

    def _tokenize_corpus(self, corpus: str) -> List[str]:
        tokenized_corpus = corpus.split()
        return tokenized_corpus

    def _init_vocabulary(self, tokens: List[str]):
        self._vocabulary = set(tokens)
