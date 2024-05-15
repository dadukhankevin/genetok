class RelativeTokenizer:
    def __init__(self, context_size):
        self.context_size = context_size

    def tokenize(self, text):
        words = text.split()
        token_to_value = {'<EOP>': 0}
        value_to_token = {0: '<EOP>'}
        next_value = 1
        relative_tokens = []
        for word in words:
            if word not in token_to_value:
                token_to_value[word] = next_value
                value_to_token[next_value] = word
                next_value += 1
            relative_tokens.append(token_to_value[word])
        return relative_tokens, token_to_value, value_to_token

    def detokenize(self, relative_tokens, value_to_token):
        try:
            words = [value_to_token[value] for value in relative_tokens]
        except:
            return 'unk'
        return ' '.join(words)

    def generate_input_output_pairs(self, text):
        input_output_pairs = []
        words = text.split()
        for i in range(len(words) - self.context_size):
            context = words[i:i + self.context_size]
            output_words = words[i + 1:i + self.context_size + 1]
            relative_tokens, token_to_value, value_to_token = self.tokenize(' '.join(context))
            input_seq = relative_tokens
            output_seq = [token_to_value.get(word, token_to_value['<EOP>']) for word in output_words]
            input_output_pairs.append((input_seq, output_seq))
        return input_output_pairs