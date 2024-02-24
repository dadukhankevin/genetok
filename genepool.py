import random
from .layers import RangeToken
from Finch import genepools, layers


class RangePool(genepools.GenePool):
    def __init__(self, min_range, max_range, source_text: str, right_freezable: bool = False, left_freezable: bool = True):
        super().__init__(len(source_text), 'cpu')
        self.source_text = source_text
        self.min_range = min_range
        self.max_range = max_range
        self.right_freezable = right_freezable
        self.left_freezable = left_freezable

    def generate_individual(self):
        index = random.randint(0, self.length - self.max_range)

        _range = random.randint(self.min_range, self.max_range)
        return RangeToken(start=index, end=index + _range, source=self.source_text, gene_pool=self,
                          right_freezable=self.right_freezable, left_freezable=self.left_freezable)
