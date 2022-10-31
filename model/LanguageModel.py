

class Ngram(object):
    def __init__(self, corpus, gram_vocab_size=None, mini_count=None, n=3):
        self.n = n
        self.count_n = {}
        self.count_n_1 = {}
        self.epsilon = 1e-4
        for s in corpus:
            wList = ['<pad>'] * (n-1) + s.split()
            for i in range(n, len(wList)):
                gram = '#'.join(wList[i-n:i])
                gram_1 = '#'.join(wList[i-n:i-1])
                self.count_n[gram] = self.count_n.get(gram, 0) + 1
                self.count_n_1[gram_1] = self.count_n_1.get(gram_1, 0) + 1
        if mini_count:
            self.count_n = {g: n for g, n in self.count_n.items() if n >= mini_count}
            self.count_n_1 = {}
            for k, v in self.count_n.items():
                assert len(k.split('#')) == 3
                gram_1 = '#'.join(k.split('#')[:2])
                self.count_n_1[gram_1] = self.count_n_1.get(gram_1, 0) + v
        if gram_vocab_size:
            count_n_list = list(self.count_n.items())
            count_n_list.sort(key=lambda x: x[1], reverse=True)
            self.count_n = {g: n for g, n in count_n_list[:gram_vocab_size]}
            self.count_n_1 = {}
            for k, v in self.count_n.items():
                assert len(k.split('#')) == 3
                gram_1 = '#'.join(k.split('#')[:2])
                self.count_n_1[gram_1] = self.count_n_1.get(gram_1, 0) + v

    def score(self, sentence):
        wList = ['<pad>'] * (self.n - 1) + sentence.split()
        p = 1
        for i in range(self.n, len(wList)):
            gram = '#'.join(wList[i-self.n: i])
            gram_1 = '#'.join(wList[i-self.n: i-1])
            cn = self.count_n.get(gram, self.epsilon)
            cn_1 = self.count_n.get(gram_1, self.epsilon)
            assert cn <= cn_1
            p *= cn/cn_1
        return p
