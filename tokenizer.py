import math

import numpy as np
from Finch.layers import Populate, CapPopulation, SortByFitness
from Finch.environments import Sequential
from .layers import *
from .genepool import *
import pickle
from tqdm import tqdm


# todo identify gaps based on the halving rule
class GeneticTokenizer:
    def __init__(self, threshold=.001, min_range=2, max_range=6, max_population=11, start_population=10, mutate_amount=5, famileis=2, step_epochs: int = 1, existing_tokens: list = []):
        self.fitness_results = {}  # for speed boost
        self.tokens = existing_tokens
        self.step_epochs = step_epochs
        self.min_range = min_range
        self.max_range = max_range
        self.last_iteration = 0
        self.max_population = max_population
        self.start_population = start_population
        self.threshold = threshold
        self.mutate_amount = mutate_amount
        self.families = famileis

    def evolve(self, dataset):
        total = len(dataset)
        with tqdm(total=total, desc="Evolving tokenizer") as pbar:
            for text in dataset:
                self.step(text)
                pbar.update(1)

    def step(self, text: str):
        pool = RangePool(min_range=self.min_range, max_range=self.max_range, source_text=text)

        max_population = self.max_population
        start_population = self.start_population

        # Create the environment
        environment = Sequential(
            layers=[
                Populate(gene_pool=pool, population=start_population),
                MutateToken(individual_selection=self.mutate_amount),
                ParentToken(families=self.families, gene_pool=pool),
                SortByFitness(),
                CapPopulation(max_population)
            ]
        )
        environment.compile(self.fitness, verbose_every=False)
        environment.iteration = self.last_iteration
        environment.evolve(self.step_epochs)
        self.last_iteration = environment.iteration

    def fitness(self, individual: RangeToken):
        token = individual.token
        if token in self.fitness_results:
            return self.fitness_results[token]

        source_text = individual.source
        count = source_text.count(token)
        percent = count / individual.length

        if percent > self.threshold:
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

    def interface(self):
        while 1:
            toks = self.tokenize(input("tokens: "))
            print("tokens: ", toks)
            print("detokens: ", self.detokenize(toks))

    def save(self, filename):
        """
        Save the state of the GeneticTokenizer object to a file.
        """
        with open(filename + ".gentok", 'wb') as f:
            pickle.dump({
                'fitness_results': self.fitness_results,
                'tokens': self.tokens,
                'step_epochs': self.step_epochs,
                'last_iteration': self.last_iteration
            }, f)

    def load(self, filename):
        """
        Load the state of the GeneticTokenizer object from a file and return a new instance.
        """

        with open(filename + ".gentok", 'rb') as f:
            data = pickle.load(f)

        self.tokens = data['tokens']
        self.fitness_results = data['fitness_results']
        self.last_iteration = data['last_iteration']


