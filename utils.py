import numpy as np
from typing import List, Any

PUNCTUATION = '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~‘’“'
EOS = {".": " <eos> ", "?": " <eos> ", "!": " <eos> "}
TRANSLATE = {**EOS, **{p: "" for p in PUNCTUATION}}

MODEL_DIRECTORY = "./trained_models/"


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def flatten(list: List[List[Any]]) -> List[Any]:
    return [j for i in list for j in i]


if __name__ == "__main__":
    print(TRANSLATE)
