import os
import numpy as np
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


if __name__ == '__main__':
    trainList, index2Word, word2Index = loadTrainingData()
    # print(len(word2Index), len(index2Word))

    p = 1.0 / len(STATUS)
    pi = [p] * len(STATUS)

    hmm = Hmm([], [], pi, word2Index, index2Word)
    A = hmm.getAExpected(trainList)
    B = hmm.getBExpected(trainList)
    pi = hmm.getPiExpected(trainList)
    print(pi)

