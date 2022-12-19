import numpy as np
class PseudoReranker(object):
    def __init__(self, accs, consistencies):
        self.accs = accs
        self.consistencies = consistencies

    def __call__(self, rank1, rank2):
        correct_answer = rank1 < rank2
        answer = np.random.choice([correct_answer, not correct_answer], p=[self.accs[abs(rank1-rank2)], 1 - self.accs[abs(rank1-rank2)]])

