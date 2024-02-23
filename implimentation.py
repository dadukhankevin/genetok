from datasets import load_dataset
from tqdm import tqdm
from genetok.tokenizer import GeneticTokenizer

# Load the dataset
dataset = load_dataset("JeanKaddour/minipile")
train_data = dataset["train"]
test_data = dataset["test"][0]['text']  # Taking only the first text for testing

# Processing the training data
print("Processing dataset...")
data = []
combined_text = ""
for sentence in tqdm(train_data, total=len(train_data), desc="Processing dataset", colour="green"):
    combined_text += sentence['text']
    if len(combined_text) > 10000:
        data.append(combined_text)
        combined_text = ""

# Tokenizing
print("Tokenizing...")
tokenizer = GeneticTokenizer(4)
tokenizer.evolve(data)
# Optionally save or load tokenizer
# tokenizer.save("tokenizer")
# tokenizer.load("tokenizer")

# Using the tokenizer interface
tokenizer.interface()
