# GeneTok: Genetic Algorithm-based Tokenizer

GeneTok is a Python library that employs genetic algorithms to craft a tokenizer. This method stands out by using the principles of genetic evolution, the concepts of individuals (in this case, index ranges from the text), mutation, and crossover—to dynamically generate and refine token sets based on the input text(s). This approach is especially beneficial for natural language processing (NLP) tasks, offering a novel solution where traditional tokenization methods might be slow.

## Features

- **Genetic Algorithm Foundation**: Built on the [Finch](https://github.com/dadukhankevin/Finch) library, GeneTok excels in speed and efficiency, utilizing genetic algorithms for token evolution.
- **Customizable Tokenization**: Users can define token size ranges and control the token evolution process, allowing for tailored tokenization strategies.
- **Fitness Function Optimization**: Utilizes a fitness function to assess and select the most effective tokens, considering their frequency and relevance in the source text.
- **Serialization Support**: Enables saving and loading the tokenizer's state, facilitating easy reuse and distribution.
- **Resumable Training**: Training sessions can be paused and resumed with entirely different texts, offering flexibility in model development.


## Colab notebooks:
- **simple example**: [genetok](https://colab.research.google.com/drive/1l2C2ruaRv2dZSmBanT9zgMET7kypIJ4u?usp=sharing). Quick overview of the library, train a tokenizer on a few GBs of text rather quickly,
## Installation

GeneTok requires Python 3.6 or later. You can install GeneTok directly from the source code:
```bash
pip install genetok
```
## Quick Start

Please look at the Colab notebooka

## How It Works

Genetok uses a genetic algorithm to evolve a set of tokens that are most effective for tokenizing a given text. It starts with a random set of tokens and 
iteratively applies genetic operations such as mutation and crossover to evolve these tokens. Each token is represented simply by it's start and end index in a source text. Mutation causes these ranges to change. Every time a good token is found it is added to the list. The fitness of each token is determined based on its 
frequency and utility in the source text, guiding the selection process towards more effective tokenization strategies.

## Drawbacks:
- Speed has it's costs, the tokens may not be the absolute global "best", but the training is *much* faster than typical tokenizers. 
- Far from complete, lots more features to add and bugs to weed out.

## Example Implementation

For a detailed example of how to use Genetok on a larger dataset, refer to `implimentation.py` in the repository. This example demonstrates loading a 
dataset, processing the text, evolving the tokenizer, and then using it to tokenize new texts.
