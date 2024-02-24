import random
from Finch import genepools, layers
from Finch.genetics import Individual
from typing import Union


class RangeToken(Individual):
    def __init__(self, gene_pool: genepools.GenePool, start: int, end: int, source: str):
        super().__init__(gene_pool=gene_pool, genes=None, fitness=1.0)
        self.start = start
        self.end = end
        self.source = source
        self.token = self.source[self.start:self.end]
        self.age = 0
        self.length = len(self.source)
        self.right_frozen = False
        self.left_frozen = False
        try:
            while self.token[0] == " ":
                self.start += 1
                self.token = self.token[1:-1]
                self.left_frozen = True
        except IndexError:
            pass
    def copy(self):
        return RangeToken(self.gene_pool, self.start, self.end, self.source)
    def mutate(self):
        choice = random.choice(['shift', 'length'])
        # if self.age > 10:
        #     self.redo()

        self.age += 1
        if choice == 'shift':
            if self.end < self.length + 1:
                shift = random.choice([1, -1])
                if not self.left_frozen:
                    self.start += shift
                self.end += shift
        if choice == "length":
            if self.end < self.length + 1:
                position = random.choice(['start', 'end'])
                if position == 'start' and not self.left_frozen:
                    self.start += random.choice([-1, 1])
                if position == 'end' or self.left_frozen:
                    self.end += random.choice([-1, 1])

        if self.end > self.start:
            self.end, self.start = self.start, self.end

        self.token = self.source[self.start:self.end]
        try:
            if self.token[0] == " ":
                self.start += 1
                self.token = self.token[1:-1]
                self.left_frozen = True
        except IndexError:
            pass
        # if self.token[-1] == " ":
        #     self.right_frozen True
    def redo(self):
        new = self.gene_pool.generate_individual()
        self.start = new.start
        self.end = new.end
        self.token = new.token
        self.left_frozen = new.left_frozen
        self.right_frozen = new.left_frozen
        self.age = 0

        try:
            while self.token[0] == " ":
                self.start += 1
                self.token = self.token[1:-1]
                self.left_frozen = True
        except IndexError:
            pass





def create_rangetoken(token: str, source_text: str, gene_pool: genepools.GenePool) -> Union[RangeToken, None]:
    try:
        start = source_text.index(token)
        end = start + len(token)
        token = RangeToken(gene_pool=gene_pool, start=start, end=end, source=source_text)
        return token
    except ValueError:
        return None
class MutateToken(layers.layer.Mutate):
    def __init__(self, individual_selection, refit: bool = True):
        super().__init__(individual_selection=individual_selection, refit=refit, gene_selection=None)
        self.selection = individual_selection

    def mutate_one(self, individual, environment):
        individual.mutate()


class ParentToken(layers.layer.Parent):
    def __init__(self, families: int, gene_pool: genepools.GenePool, refit=True):
        super().__init__(families=families, children=2, refit=refit, track_genealogies=True)
        self.gene_pool = gene_pool

    def parent(self, parent1: RangeToken, parent2: RangeToken, environment) -> list:
        a_token = parent1.token
        b_token = parent2.token

        a_child = a_token + b_token
        b_child = b_token + a_token

        a_child = create_rangetoken(a_child, source_text=parent1.source, gene_pool=self.gene_pool)
        b_child = create_rangetoken(b_child, source_text=parent2.source, gene_pool=self.gene_pool)

        return [child for child in [a_child, b_child] if child]
