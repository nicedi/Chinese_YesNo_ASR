# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


class BRNN(chainer.Chain):
    def __init__(self, n_infeature, n_units, n_outfeature, train):
        super(BRNN, self).__init__(
            fh_unit=L.LSTM(n_infeature, n_units),
            bh_unit=L.LSTM(n_infeature, n_units),
            fo_unit=L.Linear(n_units, n_outfeature),
            bo_unit=L.Linear(n_units, n_outfeature),
        )
        self.train = train

    def reset_state(self):
        self.fh_unit.reset_state()
        self.bh_unit.reset_state()

    def __call__(self, input_seq):
        seq_length = len(input_seq)
        f_layer = [self.fh_unit(x) for x in input_seq]
        b_layer = [self.bh_unit(input_seq[i]) for i in range(seq_length-1, -1, -1)]

        fo_layer = [self.fo_unit(x) for x in f_layer]
        bo_layer = [self.bo_unit(b_layer[i]) for i in range(seq_length-1, -1, -1)]

        return [fo_layer[i]+bo_layer[i] for i in range(seq_length)]
        #return [F.dropout(fo_layer[i]+bo_layer[i], train=self.train) for i in range(seq_length)]



class RNNASR(chainer.Chain):
    def __init__(self, n_feature, n_units, n_symbols, train=True):  # train flag related to Dropout
        super(RNNASR, self).__init__(
            l1=BRNN(n_feature, n_units, 32, train=train),
            l2=BRNN(32, n_units, n_symbols, train=train), 
        )

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return h
