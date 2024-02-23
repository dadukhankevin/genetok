from datasets import load_dataset
from genetok import GeneticTokenizer

dataset = load_dataset("JeanKaddour/minipile")
train = dataset["train"]
# Create an iterator from the dataset
data = []
last = ""
length = len(train)
current = 0
for sentence in train:
    current += 1
    if current % 500 == 0:
        print(current/length)
    last += sentence['text']
    if len(last) > 10000:
        data.append(last)
        last = ""
    if current > 10000:
        break

test = dataset["test"]
test_data = ""
for i in range(1):
    test_data += test[i]['text']

print("tokenizing")
tokenizer = GeneticTokenizer(2)

tokenizer.evolve(data)
tokenizer.interface()