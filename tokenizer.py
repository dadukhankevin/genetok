import math

import numpy as np
from Finch.layers import Populate, CapPopulation, SortByFitness
from Finch.environments import Sequential
from .layers import *
from .genepool import *
import pickle
from tqdm import tqdm


class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_index = -1  # -1 indicates no token ends here
        self.is_end_of_token = False  # Indicates if a complete token ends at this node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token, index):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_index = index
        node.is_end_of_token = True  # Mark the end of a complete token

    def search(self, text):
        tokens = []
        node = self.root
        start_index = 0  # Start index of the current token being processed

        while start_index < len(text):
            node = self.root  # Reset node to root for each new starting character
            longest_token_index = -1  # Initialize with -1 indicating no token found yet
            longest_token_length = 0  # Length of the longest token found

            for i in range(start_index, len(text)):
                char = text[i]
                if char in node.children:
                    node = node.children[char]
                    if node.is_end_of_token:
                        longest_token_index = node.token_index  # Update if a longer token is found
                        longest_token_length = i - start_index + 1
                else:
                    break  # Break the loop if current character is not in children

            if longest_token_index != -1:
                tokens.append(longest_token_index)  # Append the longest token index found
                start_index += longest_token_length  # Move start index to the end of the longest token
            else:
                start_index += 1  # Move to the next character if no token was found

        return tokens


# todo identify gaps based on the halving rule
class GeneticTokenizer:
    def __init__(self, threshold=.001, min_range=2, max_range=6, max_population=11, start_population=10, mutate_amount=5,
                 families=2, step_epochs: int = 1, existing_tokens: list = [], right_freezable=False, left_freezable=True):
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
        self.families = families
        self.trie = Trie()
        self.right_freezable = right_freezable
        self.left_freezable = left_freezable
        for i, token in enumerate(self.tokens):
            self.trie.insert(token, i)  # Populate Trie with existing tokens


    def evolve(self, dataset):
        total = len(dataset)
        with tqdm(total=total, desc="Evolving tokenizer") as pbar:
            for text in dataset:
                self.step(text)
                pbar.update(1)

    def step(self, text: str):
        pool = RangePool(min_range=self.min_range, max_range=self.max_range,
                         source_text=text, right_freezable=self.right_freezable, left_freezable=self.left_freezable)

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
                # Update the trie with the new token
                self.trie.insert(token, len(self.tokens) - 1)
        score = len(token) * percent
        self.fitness_results.update({token: score})

        return score

    def tokenize(self, text):
        indices = self.trie.search(text)
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
        Save the state of the GeneticTokenizer object to a file, including the Trie.
        """
        with open(filename + ".gentok", 'wb') as f:
            pickle.dump({
                'fitness_results': self.fitness_results,
                'tokens': self.tokens,
                'trie': self.trie,  # Save the Trie structure
                'min_range': self.min_range,
                'max_range': self.max_range,
                'max_population': self.max_population,
                'start_population': self.start_population,
                'threshold': self.threshold,
                'mutate_amount': self.mutate_amount,
                'families': self.families,
                'right_freezable': self.right_freezable,
                'left_freezable': self.left_freezable,
            }, f)

    def load(self, filename):
        """
        Load the state of the GeneticTokenizer object from a file, including the Trie.
        """
        with open(filename + ".gentok", 'rb') as f:
            data = pickle.load(f)

        self.tokens = data['tokens']
        self.fitness_results = data['fitness_results']
        self.trie = data['trie']  # Load the Trie structure

        # Load additional attributes
        self.min_range = data['min_range']
        self.max_range = data['max_range']
        self.max_population = data['max_population']
        self.start_population = data['start_population']
        self.threshold = data['threshold']
        self.mutate_amount = data['mutate_amount']
        self.families = data['families']
        self.right_freezable = data['right_freezable']
        self.left_freezable = data['left_freezable']
