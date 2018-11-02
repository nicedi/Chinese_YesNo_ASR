# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers
import chainer.functions as F  # F.ctc
from model import RNNASR
import utils
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import matplotlib.font_manager as mfm
font_path = "SimHei.ttf"
prop = mfm.FontProperties(fname=font_path)


# https://github.com/jameslyons/python_speech_features
from python_speech_features import mfcc
from wer import wer


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


def get_mean_std(train_set):
    total = None
    for item in train_set:
        samplerate, wavdata = wav.read(os.path.join(data_root, item[0]))
        feats = mfcc(wavdata, samplerate).astype(np.float32)
        if total is None:
            total = feats
        else:
            total = np.vstack((total, feats))
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return mean.reshape((1, -1)), std.reshape((1, -1))


## Train
def forward_one_sample(model, wavfile, label, SIL_idx, useGPU):
    try:
        samplerate, wavdata = wav.read(wavfile)
    except IOError:
        return None, None
    feats = mfcc(wavdata, samplerate).astype(np.float32)
    feats = (feats - mean) / std

    model.reset_state()

    if useGPU:
        input_seq = [Variable(cuda.to_gpu(feats[i, :].reshape((1, -1))))
                     for i in range(feats.shape[0])]
        y = model(input_seq)
        label = Variable(cuda.to_gpu(
            xp.array(label, dtype=xp.int32).reshape((1, -1))))
    else:
        input_seq = [Variable(feats[i, :][np.newaxis, :])
                     for i in range(feats.shape[0])]
        # y = [model(item) for item in input_seq]
        y = model(input_seq)
        label = Variable(xp.array(label, dtype=xp.int32).reshape((1, -1)))

    loss = F.connectionist_temporal_classification(y, label, SIL_idx)
    return y, loss


def plot_ctc(y):
    y = np.array(y).squeeze()
    p = plt.plot(y)
    plt.axis([0, 250, -0.1, 1.1])
    plt.legend(p, ['blank', '可', '以', '不', '行'], prop=prop)
    plt.show()


# Evaluate on test dataset
def evaluate(testset):
    evaluator = model.copy()  # to use different state
    evaluator.train = False
    # for item in testset:
    total_symbol = 0
    error_symbol = 0
    for item in testset:
        evaluator.reset_state()   # initialize state 是否重置关系不大

        x_data = os.path.join(data_root, item[0])
        y_data = item[1]
        y, _ = forward_one_sample(evaluator, x_data, y_data, SIL_idx, useGPU)
        if y is None:
            continue
        # decoding
        y_prob = [F.softmax(y[i]).data for i in range(len(y))]

        # observe the model output by uncommenting the following line
        plot_ctc(y_prob)

        y_dec = [y_prob[i].argmax() for i in range(len(y))]
        num_seq = utils.compress_seq(y_dec, SIL_idx)
        print('decode sequence: ', num_seq)
        print('target sequence: ', y_data)
        total_symbol += len(y_data)
        error_symbol += wer(y_data, num_seq)
    print('WER: ', str(float(error_symbol) / total_symbol * 100) + '%')


if __name__ == '__main__':
    ## Prepare dataset
    data_root = "yesno_cn"
    save_head = "yesno_cn"

    train_list = os.path.join(data_root, "train_list.txt")
    dataset = []
    with open(train_list, 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            # line = line.decode('utf-8')
            wavfile, transcribe = line.split(' ', 1)
            if transcribe.strip() == '可以':
                label = [1, 2]
            else:
                label = [3, 4]

            dataset.append([wavfile, label])

    np.random.seed(0)
    np.random.shuffle(dataset)
    trainset = dataset[ : 27]
    testset = dataset[27 : ]
    mean, std = get_mean_std(trainset)

    ## Prepare model
    n_feature = 13
    n_units = 300
    n_symbol = 5
    n_epoch = 2  # number of epochs
    SIL_idx = 0  # index of blank symbol
    grad_clip = 10   # gradient norm threshold to clip 较大的值，模型收敛的较快

    model = RNNASR(n_feature, n_units, n_symbol)

    ## use GPU or not
    useGPU = False
    xp = cuda.cupy if useGPU else np
    if useGPU:
        cuda.get_device(0).use()
        model.to_gpu()

    ## Setup optimizer
    optimizer = optimizers.NesterovAG() # 比RMSpropGraves快一些
    #optimizer = optimizers.RMSpropGraves()

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip)) # 必不可少的

    # train loop
    counter = 0
    trainsize = len(trainset)
    for epoch in range(1, n_epoch+1):
        indexes = np.random.permutation(trainsize)
        for i in range(trainsize):

            x_data = os.path.join(data_root, trainset[indexes[i]][0])
            y_data = trainset[indexes[i]][1]

            _, loss = forward_one_sample(model, x_data, y_data, SIL_idx, useGPU)
            if loss is None: # for file missing problem
                continue

            print('epoch %d %d of %d loss: %.4f' % (epoch, i, trainsize, loss.data))

            model.cleargrads()
            loss.backward()
            optimizer.update()

        ## Save the model and the optimizer and evaluate model
        # print('save the model')
        # serializers.save_npz(save_head+str(epoch)+'.model', model)
        # print('save the optimizer')
        # serializers.save_npz(save_head+str(epoch)+'.state', optimizer)
        evaluate(testset)




