import numpy as np
from Finch.layers import Populate, CapPopulation, SortByFitness
from Finch.environments import Sequential
from layers import *
from genepool import *
from typing import Union
import string



## todo identify gaps based on the halving rule
class GeneticTokenizer:
    def __init__(self, step_epochs: int = 1, existing_tokens: list = []):
        self.fitness_results = {}  # for speed boost
        self.tokens = existing_tokens
        self.step_epochs = step_epochs
        self.last_iteration = 0

    def evolve(self, dataset):
        total = len(dataset)
        n = 0
        for text in dataset:
            n += 1
            self.step(text)
            if n % 2000 == 0:
                print(f"step: {n} of {total}")


    def step(self, text: str):
        pool = RangePool(min_range=1, max_range=6, source_text=text)

        max_population = 10
        start_population = 11

        ## Create the environment

        environment = Sequential(
            layers=[
                Populate(gene_pool=pool, population=start_population),
                MutateToken(individual_selection=5),
                ParentToken(families=2, gene_pool=pool),
                SortByFitness(),
                CapPopulation(max_population)
            ]
        )
        environment.compile(self.fitness, verbose_every=1)
        environment.iteration = self.last_iteration
        environment.evolve(self.step_epochs)
        self.last_iteration = environment.iteration

    def fitness(self, individual: RangeToken):
        token = individual.token
        if token in self.fitness_results:
            return self.fitness_results[token]

        source_text = individual.source
        count = source_text.count(token)
        if count > 5:
            if token not in self.tokens:
                self.tokens.append(token)
        self.fitness_results.update({token: count})
        return count

    def tokenize(self, text):
        """
        Tokenize the given text into indices of tokens from the global `tokens` list.
        """
        indices = []
        i = 0
        input(len(text))
        while i < len(text):
            matched = False
            for token in sorted(self.tokens, key=len, reverse=True):  # Sort tokens by length, descending
                if text[i:].startswith(token):
                    indices.append(self.tokens.index(token))
                    i += len(token)
                    matched = True
                    break
            if not matched:
                i += 1  # Move to the next character if no token matches
        return indices

    def detokenize(self, indices):
        """
        Detokenize the given list of indices into the original text.
        """
        return '|'.join(self.tokens[i] for i in indices)
    def narrow(self, text):
        characters = np.unique(np.asarray(list(text)))
        self.tokens.extend(characters)
        print("length: ", len(self.tokens))

        tokens = self.tokenize(text)
        self.tokens = np.asarray(self.tokens)
        unique = np.unique(np.asarray(tokens))
        self.tokens = self.tokens[unique].tolist()
        print("length: ", len(self.tokens))


    def interface(self):
        while 1:
            toks = self.tokenize(input("tokens: "))
            print("tokens: ", toks)
            print("detokens: ", self.detokenize(toks))

