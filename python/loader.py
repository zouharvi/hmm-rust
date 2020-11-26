class Mapper:
    def __init__(self):
        self.map_to = {}
        self.map_from = {}
        self.counter = 0

    def clone(self):
        new = Mapper()
        new.map_to = dict(self.map_to)
        new.map_from = dict(self.map_from)
        new.counter = self.counter
        return new

    def update(self, tok):
        if tok in self.map_to:
            return self.map_to[tok]
        else:
            self.map_to[tok] =  self.counter
            self.map_from[self.counter] = tok
            self.counter += 1
            return self.counter - 1
    
    def count(self):
        if self.counter == 0:
            return None
        else:
            return self.counter

class Sentence:
    def __init__(self):
        self.tokens = []

class Loader:
    def __init__(self, loader, path):
        self.data = []
        if loader == None:
            self.mapper_w = Mapper()
            self.mapper_t = Mapper()
        else:
            self.mapper_w = loader.mapper_w.clone()
            self.mapper_t = loader.mapper_t.clone()

        with open(path, 'r') as f:
            sent = Sentence()
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if len(sent.tokens) != 0:
                        self.data.append(sent)
                    sent = Sentence()
                else:
                    vals = line.split("\t")
                    sent.tokens.append((self.mapper_w.update(vals[0]), 0 if len(vals) == 1 else self.mapper_t.update(vals[1])))