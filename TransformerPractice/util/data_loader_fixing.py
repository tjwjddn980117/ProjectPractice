from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.datasets.translation import Multi30k

class DataLoader:
    source: get_tokenizer = None
    target: get_tokenizer = None

    def __init__(self, ext, tokenize_en, tokenize_de):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        print('dataset initializing start')
    
    def yield_tokens(self, data_iter, language):
        if language == 'de':
            tokenize = self.tokenize_de
        elif language == 'en':
            tokenize = self.tokenize_en
        else:
            raise ValueError("Unsupported language: " + language)

        for _, text in data_iter:
            yield tokenize(text)
    
    def process_data(self, data, source_vocab, target_vocab):
        processed_data = []
        for example in data:
            src = example.src
            trg = example.trg
            src_tokens = [source_vocab[token] for token in src]
            trg_tokens = [target_vocab[token] for token in trg]
            processed_data.append((src_tokens, trg_tokens))
        return processed_data

    def make_dataset(self):
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext)

        if self.ext == ('.de', '.en'):
            self.source = build_vocab_from_iterator(self.yield_tokens(train_data, 'de'), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
            self.target = build_vocab_from_iterator(self.yield_tokens(train_data, 'en'), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        elif self.ext == ('.en', '.de'):
            self.source = build_vocab_from_iterator(self.yield_tokens(train_data, 'en'), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
            self.target = build_vocab_from_iterator(self.yield_tokens(train_data, 'de'), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        
        self.source.set_default_index(self.source['<unk>'])
        self.target.set_default_index(self.target['<unk>'])

        train_data = self.process_data(train_data, self.source, self.target)
        valid_data = self.process_data(valid_data, self.source, self.target)
        test_data = self.process_data(test_data, self.source, self.target)
        return train_data, valid_data, test_data, self.source, self.target
    
    # not a recursive function.
    # For each of the source and target, a vocabulary book containing only words with a minimum frequency of min_freq or more is generated based on training data.
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)
    
    # BucketIterator is a repeater for processing sequence data. 
    # BucketIterator creates a mini-batch that uses minimal padding by grouping examples of similar lengths.
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = build_vocab_from_iterator.splits((train, validate, test),
                                                                              batch_sizes=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator