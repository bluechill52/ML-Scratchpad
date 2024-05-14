import os
import re



class SimpleTokenizer:
    def __init__(self, fname):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        tokens = self.tokenize(raw_text)
        self.vocab = sorted(list(set(tokens)))
        self.encode_map = {token : idx for idx, token in enumerate(self.vocab)}
        self.decode_map = {idx : token for idx, token in enumerate(self.encode_map)}
        self.n_vocab = len(self.vocab)
        
    def tokenize(self, text):
        # Tokenize on punctuations and whitespace
        splits = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [split for split in splits if split.strip()]
        return tokens
        
    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self.encode_map[token] for token in tokens if token.strip()]
        return ids
    
    def decode(self, ids):
        tokens = [self.decode_map[id] for id in ids]
        return ' '.join(tokens)
    
    def getVocabSize(self):
        return len(self.vocab)
        

if __name__ == '__main__':
    fname = 'the-verdict.txt'
    context_size = 5
    
    Tokenizer = SimpleTokenizer(fname)
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    ids = Tokenizer.encode(raw_text)
    
    assert len(ids) == len(Tokenizer.tokenize(raw_text))
    print(ids)
    