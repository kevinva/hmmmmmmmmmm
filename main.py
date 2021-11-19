import os
import time
import numpy as np
from numpy.core.fromnumeric import transpose
from hmm import STATUS, Hmm


def loadTrainingData():
    # filePath = './icwb2-data/training/msr_training.utf8'
    # with open(filePath, 'r', encoding='UTF-8') as file:
    #     trainingList = list()
    #     lines = file.readlines()[:1]   # hoho_test
    #     for line in lines:
    #         words = line.split('  ')
    #         words = words[1:-1]  # 去掉开头的双引号，去掉最后的换行符  hoho_test
    #         if len(words) == 0:
    #             continue

    #         trainingList.append(words)
    # return trainingList

    words = ['人们', '常', '说', '生活', '是',  '一',  '部',  '教科书']
    trainingList = list()
    trainingList.append(words)
    return trainingList

def getWordsInfo(trainingList):
    index2Word = list()
    word2Index = dict()

    for words in trainingList:
        sentence = ''.join(words)
        print(sentence)
        for word in sentence:
            if word not in index2Word:
                word2Index[word] = len(index2Word)
                index2Word.append(word)

    return index2Word, word2Index


def probabilityEstimationProblemTest(trainList, index2Word, word2Index):
    pi = Hmm.getPiExpected(trainList)
    A = Hmm.getAExpected(trainList)
    B = Hmm.getBExpected(trainList, index2Word, word2Index)
    print('pi: ', pi) 
    print('A: ', A)
    print('B: ', B)
       
    hmm = Hmm(A, B, pi, word2Index, index2Word)

    observation = trainList[0]
    sentence = ''.join(observation)
    print('观测序列: "{}"'.format(sentence))

    startTime = time.time()
    pOS = hmm.forward(sentence)
    endTime = time.time()
    print(' （前向, 递归计算）P(O|θ): {}'.format(pOS))
    print(' （前向, 递归计算）计算耗时: {}s'.format(endTime - startTime))
    
    startTime = time.time()
    pOS = hmm.forwardv2(sentence)
    endTime = time.time()
    print(' （前向, 矩阵计算）P(O|θ): {}'.format(pOS))
    print(' （前向, 矩阵计算）计算耗时: {}s'.format(endTime - startTime))

    startTime = time.time()
    pOS = hmm.backward(sentence)
    endTime = time.time()
    print(' （后向, 递归计算）P(O|θ): {}'.format(pOS))
    print(' （后向, 递归计算）计算耗时: {}s'.format(endTime - startTime))

    startTime = time.time()
    pOS = hmm.backwardv2(sentence)
    endTime = time.time()
    print(' （后向, 矩阵计算）P(O|θ): {}'.format(pOS))
    print(' （后向, 矩阵计算）计算耗时: {}s'.format(endTime - startTime))


def learningProblemTest(trainList, index2Word, word2Index):
    # N = len(STATUS)
    # C = len(index2Word)
    # pi = [1.0 / N] * N
    # A = np.full((N, N), 1.0 / N)
    # B = np.full((N, C), 1.0 / C)

    pi = Hmm.getPiExpected(trainList)
    A = Hmm.getAExpected(trainList)
    B = Hmm.getBExpected(trainList, index2Word, word2Index)
    print('pi: ', pi) 
    print('A: ', A)
    print('B: ', B)

    hmm = Hmm(A, B, pi, word2Index, index2Word)
    hmm.fit(trainList)


if __name__ == '__main__':
    trainList = loadTrainingData()
    index2Word, word2Index = getWordsInfo(trainList)
    # print(trainList)
    # print(word2Index)
    # print(index2Word)

    # probabilityEstimationProblemTest(trainList, index2Word, word2Index)
    learningProblemTest(trainList, index2Word, word2Index)