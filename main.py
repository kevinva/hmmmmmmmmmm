import os
import numpy as np
from hz_hmm import STATUS, HzHmm


def loadTrainingData():
    filePath = './icwb2-data/training/pku_training.utf8'
    with open(filePath) as file:
        trainingList = list()
        for line in file.readlines():
            words = line.split('  ')
            words = words[:-1]  # 去掉最后的换行符
            if len(words) > 0:
                trainingList.append(words)

        print(len(trainingList))



if __name__ == '__main__':
    # loadTrainingData()
    print(len(STATUS))