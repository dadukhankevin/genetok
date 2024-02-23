# GeneTok: Genetic Algorithm-based Tokenizer

Genetok is a Python library that leverages genetic algorithms to develop a tokenizer. This innovative approach allows for the dynamic creation and optimization of token sets based on the input text, making it particularly useful for natural language processing tasks where traditional tokenization methods may fall short.

## Features

- **Genetic Algorithm**: It is quite speedy and relies on [Finch](https://github.com/dadukhankevin/Finch) for the underlying algorithm.
- **Customizable Tokenization**: Offers flexibility in defining the range of token sizes and the selection process for token evolution. 
- **Fitness Function Optimization**: Employs a fitness function to evaluate and select the most effective tokens based on their occurrence and relevance in the source text.
- **Serialization Support**: Provides functionality to save and load the state of the tokenizer, allowing for easy reuse and distribution.
- **Resumable**: You can resume training with completely different text later!

## Installation

Genetok requires Python 3.6 or later. You can install Genetok directly from the source code:
bash
git clone https://github.com/yourusername/genetok.git
cd genetok
pip install .

## Quick Start

Here's a quick example to get you started with Genetok:
```python
from genetok.tokenizer import GeneticTokenizer
# Initialize the GeneticTokenizer
tokenizer = GeneticTokenizer(step_epochs=4)
Sample text
text = "This is a sample text for the GeneticTokenizer."
# Evolve the tokenizer based on the sample text
tokenizer.evolve([text])
Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# Detokenize the tokens back to text
original_text = tokenizer.detokenize(tokens)
print("Original Text:", original_text)
```

## How It Works

Genetok uses a genetic algorithm to evolve a set of tokens that are most effective for tokenizing a given text. It starts with a random set of tokens and iteratively applies genetic operations such as mutation and crossover to evolve these tokens. The fitness of each token is determined based on its frequency and utility in the source text, guiding the selection process towards more effective tokenization strategies.

## Drawbacks:
- Speed has it's costs, the tokens may not be the absolute global "best", but the training is *much* faster than typical tokenizers. 
- Far from complete, lots more features to add and bugs to weed out.

## Example Implementation

For a detailed example of how to use Genetok on a larger dataset, refer to `implimentation.py` in the repository. This example demonstrates loading a dataset, processing the text, evolving the tokenizer, and then using it to tokenize new texts.
