import re

import numpy as np

from src.config import SearchConfig


class WordTokenizer:
    is_fast = True

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens=True, truncation=False,
                 return_offsets_mapping=False, verbose=True):
        assert truncation is False
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if add_special_tokens:
            result["input_ids"] = [1000] + result["input_ids"] + [1001]
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result


class FakeEncoder:
    max_seq_length = 512
    dimension = 4

    def __init__(self, config: SearchConfig):
        self.config = config
        self.tokenizer = WordTokenizer()
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        vectors = np.array([
            [1.0, text.lower().count("regularization"),
             text.lower().count("regression"), text.lower().count("classification")]
            for text in texts
        ], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
