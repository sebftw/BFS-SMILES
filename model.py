import re

import moses
from torch import nn

from preprocess import string_to_smiles


class BFSModel(nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel
        self.vocabulary = self.submodel.vocabulary

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, masks=None, **kwargs):
        # (Later: Change to apply the appropriate masks, to ensure validity.)
        return self.submodel.forward(*args, **kwargs)

    def string2tensor(self, *args, **kwargs):
        return self.submodel.string2tensor(*args, **kwargs)

    def tensor2string(self, *args, **kwargs):
        return self.submodel.tensor2string(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return string_to_smiles(self.submodel.sample(*args, **kwargs))


class WordVocab(moses.CharVocab):
    pattern = re.compile('[A-Z][a-z]*|.')
    # This may be used instead of a simple CharVocab. (Because BFS-SMILES allows for it)

    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(cls.pattern.findall(string))

        return cls(chars, *args, **kwargs)

    def __init__(self, chars={'0', '1', '2', '3', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S'}, *args, **kwargs):
        super().__init__(chars, *args, **kwargs)

    def string2ids(self, string, *args, **kwargs):
        super().string2ids(self.pattern.findall(string))
