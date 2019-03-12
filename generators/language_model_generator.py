from typing import Dict, Callable

from generators.generator_base import GeneratorBase
from language_models.language_model_base import LanguageModelBase
import utils


class LanguageModelGenerator(GeneratorBase):
    def __init__(self, language_model: LanguageModelBase):
        super().__init__()
        self._language_model = language_model

    def encode(self, message: str, seed_text: str = " ".join(["word"] * 100),
               n_best: int = 5, beam_width: int = 2) -> Dict[str, float]:
        print("[LanguageModelGenerator] Beam search in progress")
        message = self._format_message(message)
        message_chars = list(message)
        candidate_texts = {seed_text: 0.0}
        for char in message_chars:
            starts_with = lambda word: list(word)[0] == char
            candidate_texts = self._beam_search(candidate_texts, filter=starts_with, beam_width=beam_width)

        best_texts = utils.get_best(n_best, candidate_texts)
        best_texts = {" ".join(text.split(" ")[-len(message_chars):]): proba for text, proba in best_texts.items()}
        print("\t --done")
        return best_texts

    def _beam_search(self, candidate_texts: Dict[str, float], beam_width: int = 2,
                     filter: Callable[[str], bool] = lambda x: True) -> Dict[str, float]:
        new_candidate_texts = dict()
        for text, text_proba in candidate_texts.items():
            prediction = self._language_model.predict(text)
            prediction = {word: proba for word, proba in prediction.items() if word is None or filter(word)}

            candidate_words = utils.get_best(beam_width, prediction)
            for word, word_proba in candidate_words.items():
                new_text = text + " " + word
                new_proba = text_proba + word_proba
                new_candidate_texts[new_text] = new_proba

        return new_candidate_texts

    def _format_message(self, message: str) -> str:
        message = message.lower()
        return message


if __name__ == "__main__":
    from language_models.basic_lstm import BasicLSTM
    import numpy as np

    message = "censorship"

    seed_text = """The girls came just in time; they held him fast and tried to free his
                    beard from the line, but all in vain, beard and line were entangled fast
                    together. There was nothing to do but to bring out the scissors and cut
                    the beard, whereby a small part of it was lost. When the dwarf saw that
                    he screamed out: ‘Is that civil, you toadstool, to disfigure a man’s
                    face?
                    """

    basic_lstm = BasicLSTM("./data/grimm.txt")
    basic_lstm.train(epochs=100)

    generator = LanguageModelGenerator(basic_lstm)
    encodings = generator.encode(message=message, seed_text=seed_text, n_best=10, beam_width=2)

    print("[Best candidate texts]")
    for encoding, proba in encodings.items():
        print(f"{encoding} [log(p) = {np.round(proba, 2)}]")
