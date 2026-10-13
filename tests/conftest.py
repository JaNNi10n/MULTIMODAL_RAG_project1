import hashlib

import numpy as np
import pytest

from mmrag.index import tokenize
from mmrag.synth import generate


def hash_encoder(texts, dim=64):
    """Deterministic bag-of-words hashing encoder so tests need no model download."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    qa_path = generate(out, n_docs=6, seed=1)
    return out, qa_path
