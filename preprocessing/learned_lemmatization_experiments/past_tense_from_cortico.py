# %%
from typing import List
import retinasdk
import matplotlib.pyplot as plt
import numpy as np
import json

from preprocessing.learned_lemmatization_experiments.past_tense_from_w2v import past_tense

# %%
api_key = json.load(open("./preprocessing/learned_lemmatization_experiments/cortico_api_key.json", "r")).get("key")
liteClient = retinasdk.LiteClient(api_key)


# %%
def embed(word: str) -> List[int]:
    fingerprint = liteClient.getFingerprint(word)
    return fingerprint


def to_array(fingerprint: List[int]):
    dim = 128
    x = np.zeros(dim ** 2)
    x[fingerprint] = 1
    x = x.reshape((dim, dim))
    return x


# %%
embeddings = [[to_array(embed(word)) for word in pair] for pair in past_tense]

# %%
num_pairs = len(embeddings)
for i, pair in enumerate(embeddings):
    verb, past_tense = pair
    plt.subplot(num_pairs, 2, 2 * i + 1)
    plt.imshow(verb)
    plt.subplot(num_pairs, 2, 2 * i + 2)
    plt.imshow(past_tense)
plt.show()
