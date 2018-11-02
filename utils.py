# -*- coding: utf-8 -*-
from __future__ import print_function


def compress_seq(y, blank):
    t = []
    seq = []
    for label in y:
        if label == blank:
            if len(t) == 0:
                continue
            [seq.append(i) for i in t]
            t = []
        else:
            if len(t) == 0 or label != t[-1]:
                t.append(label)
    [seq.append(i) for i in t]
    return seq

def depack(y):
    res = []
    while True:
        if len(y) == 0:
            break
        res.insert(0, y[1])
        y = y[0]
    return res


if __name__ == '__main__':
    char2num = {u'?':0}
    num2char = {0:u'?'}
    counter = 1
    with open('hanzi_all.txt','r') as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.decode('utf-8')
        char2num[line.strip()] = counter
        num2char[counter] = line.strip()
        counter += 1
    char2num[u'_'] = counter
    num2char[counter] = u'_'
    
    y = ((((), 2), 4), 6)
    print(depack(y))

    a1 = ['a', '_', 'b', 'c', '_', '_']
    a2 = ['_', '_', 'a', '_', 'b', 'c']
    a3 = ['a', 'b', 'b', 'b', 'c', 'c']
    a4 = ['a', '_', 'b', '_', 'c', 'c']
    print(compress_seq(a1, '_'))
    print(compress_seq(a2, '_'))
    print(compress_seq(a3, '_'))
    print(compress_seq(a4, '_'))