class Vocab:
    def __init__(self):
        self.sym2num = {}
        self.num2sym = []

    def add(self, sym):
        if sym not in self.sym2num:
            self.sym2num[sym] = len(self.num2sym)
            self.num2sym.append(sym)

    def numberize(self, sym):
        return self.sym2num.get(sym, self.sym2num['<UNK>'])

    def denumberize(self, num):
        return self.num2sym[num]

    def __len__(self):
        return len(self.num2sym)

    def update(self, seq):
        for sym in seq:
            self.add(sym)

def read_data(path):
    # starts as one sentence per line text file
    # return format: list of lists, where big list is a sentence, and nested lists are of the characters in the sentence
    # example: [['s', 'e', 'n', 't', 'e', 'n', 'c', 'e', ' ', '1'], ['s', 'e', 'n', 't', 'e', 'n', 'c', 'e', ' ', '2']]
    with open(path) as f:
        return [['<BOS>'] + list(line.strip()) + ['<EOS>'] for line in f]