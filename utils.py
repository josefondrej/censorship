import numpy as np
from typing import List, Any, Dict

PUNCTUATION = '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~‘’“'
EOS = " <eos> "
EOS_TRANSLATE = {".": EOS, "?": EOS, "!": EOS}
TRANSLATE = {**EOS_TRANSLATE, **{p: "" for p in PUNCTUATION}}

MODEL_DIRECTORY = "./trained_models/"


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def flatten(list: List[List[Any]]) -> List[Any]:
    return [j for i in list for j in i]


def get_best(n: int, dictionary: Dict[Any, float]) -> Dict[Any, float]:
    sorted_keys = sorted(dictionary.keys(), key=dictionary.get, reverse=True)
    return {k: dictionary[k] for k in sorted_keys[:n]}


if __name__ == "__main__":
    print(TRANSLATE)
    dictionary = {"a": 1, "c": 10, "d": 24, "b": 3}
    best_keys = get_best(5, dictionary)
    print(best_keys)
