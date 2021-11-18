import os
import time
import numpy as np
from numpy.core.fromnumeric import transpose
from hmm import STATUS, Hmm


def loadTrainingData():
    filePath = './icwb2-data/training/pku_training.utf8'
    with open(filePath, 'r', encoding='UTF-8') as file:
        trainingList = list()
        index2Word = list()
        word2Index = dict()
        for line in file.readlines()[:100]:
            words = line.split('  ')
            words = words[:-1]  # 去掉最后的换行符
            if len(words) == 0:
                continue

            trainingList.append(words)

            for wordItems in words:   
                for word in wordItems:
                    if word not in index2Word:
                        word2Index[word] = len(index2Word)
                        index2Word.append(word)
    
    return trainingList, index2Word, word2Index

def observationProbabilityCalculationTest(trainList, index2Word, word2Index):
    A = Hmm.getAExpected(trainList)
    B = Hmm.getBExpected(trainList, index2Word, word2Index)
    pi = Hmm.getPiExpected(trainList)
    print('A: ', A)
    print('B: ', B)
    print('pi: ', pi)    
    hmm = Hmm(A, B, pi, word2Index, index2Word)

    observation = trainList[3]
    sentence = ''.join(observation)[:20]
    print('观测序列: "{}"'.format(sentence))

    startTime = time.time()
    pOS = hmm.forward(sentence)
    endTime = time.time()
    print(' （前向, 递归计算）P(O|λ): {}'.format(pOS))
    print(' （前向, 递归计算）计算耗时: {}s'.format(endTime - startTime))
    
    startTime = time.time()
    pOS = hmm.forwardv2(sentence)
    endTime = time.time()
    print(' （前向, 矩阵计算）P(O|λ): {}'.format(pOS))
    print(' （前向, 矩阵计算）计算耗时: {}s'.format(endTime - startTime))

    startTime = time.time()
    pOS = hmm.backward(sentence)
    endTime = time.time()
    print(' （后向, 递归计算）P(O|λ): {}'.format(pOS))
    print(' （后向, 递归计算）计算耗时: {}s'.format(endTime - startTime))

    startTime = time.time()
    pOS = hmm.backwardv2(sentence)
    endTime = time.time()
    print(' （后向, 矩阵计算）P(O|λ): {}'.format(pOS))
    print(' （后向, 矩阵计算）计算耗时: {}s'.format(endTime - startTime))

def baumWelchTest(trainList, index2Word, word2Index):
    N = len(STATUS)
    C = len(index2Word)
    
    pi = [1.0 / N] * N

    A = np.full((N, N), 1.0 / N)
    B = np.full((N, C), 1.0 / C)

    print(A)
    print(B)
    print(pi)

    hmm = Hmm(A, B, pi, word2Index, index2Word)
    hmm.fit(trainList)

if __name__ == '__main__':
    trainList, index2Word, word2Index = loadTrainingData()
    # print(len(word2Index), len(index2Word))

    # observationProbabilityCalculationTest(trainList, index2Word, word2Index)
    baumWelchTest(trainList, index2Word, word2Index)