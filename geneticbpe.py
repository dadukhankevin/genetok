from Finch.genetics import Individual
from Finch.layers import GenePool, Mutate, SortByFitness, CapPopulation, Populate, ParentSimple
from Finch.environments import Sequential
from Finch.layers.layer import Parent
from Finch.tools.rates import Rate
import numpy as np
import random as rd
from typing import List, Callable, Union



class GeneticMarkov:
    def __init__(self, corpus):
        self.corpus = corpus


class Tokenizer:
    def __init__(self, starter: np.ndarray, corpus: str):
        self.tokens = starter
        self.corpus = corpus


class GeneticBPE(Tokenizer):
    def __init__(self, starter: np.ndarray, corpus: str, max_population: int, start_population: int,
                 mutation_selection: Union[Callable | int], mutation_probability: Union[Callable | int],
                 parent_selection: Union[Callable | int], callbacks: list[Callable] = None):
        super().__init__(starter, corpus)
        self.max_population = max_population
        self.start_population = start_population
        self.mutation_selection = mutation_selection
        self.mutation_probability = mutation_probability
        self.parent_selection = parent_selection
        self.callbacks = callbacks

    def evolve(self, gene_pool: GenePool, epochs):
        environment = Sequential(layers=[
            Populate(gene_pool, self.start_population),
            MutateByteToken(gene_pool, token_selection=self.mutation_selection,
                            mutation_probability=self.mutation_probability, refit=True),
            ParentToken(self.parent_selection, refit=True),
            SortByFitness(),
            CapPopulation(self.max_population),
        ])
        environment.compile(fitness_function=self.fitness, callback=self.callback)
        environment.evolve(epochs)
    def callback(self, tokens, environment):
        for callback in self.callbacks:
            callback()


    def fitness(self, token: Individual):
        string_token = "".join([self.tokens[i] for i in token.genes])
        if string_token in self.tokens:
            return 0
        score = self.corpus.count(string_token)
        if score > 10:
            self.tokens = np.append(self.tokens, string_token)
        return score

    def change_corpus(self, corpus: str):
        self.corpus = corpus


class ByteToken(Individual):
    def __init__(self, gene_pool, idx):
        super().__init__(gene_pool=gene_pool, genes=idx, fitness=0)


class MutateByteToken(Mutate):
    def __init__(self, gene_pool: GenePool, token_selection, mutation_probability: float, refit=True):
        super().__init__(individual_selection=token_selection, gene_selection=lambda x: None, refit=refit)
        self.mutation_probability = mutation_probability
        self.gene_pool = gene_pool

    def mutate_one(self, individual: ByteToken, environment: Sequential):
        sample = rd.uniform(0, 1)
        if sample < self.mutation_probability:
            individual.genes[rd.randint(0, len(individual.genes)-1)] = self.gene_pool.generate_genes(1)


class ParentToken(ParentSimple):
    def __init__(self, families, refit=True):
        super().__init__(families=families, children=2, refit=refit)

    def parent(self, parent1: ByteToken, parent2: ByteToken, environment) -> list:
        return super().parent(parent1, parent2, environment)


class BPEPool(GenePool):
    def __init__(self, genetic_bpt_tokenizer: GeneticBPE):
        super().__init__(length=1)
        self.genetic_bpt_tokenizer = genetic_bpt_tokenizer
        self.items = []

    def generate_individual(self):
        """
        Generates a new individual.
        :return: genetics.Individual with genes of type int
        """
        index = self.generate_genes(1)
        index2 = self.generate_genes(1)


        return ByteToken(self, idx=np.array([index, index2]))

    def generate_genes(self, amount: int):
        return rd.randint(0, len(self.genetic_bpt_tokenizer.tokens) - 1)





path = "/Users/daniellosey/Desktop/code/personal/genetok_module/bible.txt"
with open(path, "r") as f:
    text = f.read().lower()

chars = "qwertyuiopasdfghjklzxcvbnm"
epochs = 100
population = Rate(50, 4000, return_int=True, epochs=100)
tokenizer = GeneticBPE(starter=np.array(list(chars)),
                       corpus=text,
                       mutation_probability=.3, max_population=50000, start_population=population.get, mutation_selection=20,
                       parent_selection=10, callbacks=[population.next])
pool = BPEPool(tokenizer)
tokenizer.evolve(pool, epochs=epochs)
print(tokenizer.tokens)
