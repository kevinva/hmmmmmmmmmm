import numpy as np

STATUS = {0, 1}   # 状态集合，0表示非结尾字，1表示结尾字

class Hmm:
    def __init__(self, A, B, pi, word2Index, index2Word):
        self.A = A      # A为状态转移概率矩阵
        self.B = B      # B为观察生成概率矩阵
        self.pi = pi    # pi为初始状态矩阵

        self.wordToIndex = word2Index      # 字典库
        self.indexToWord = index2Word

    def getPiExpected(self, trainingList):
        statusCount = len(STATUS)
        piExpect = np.zeros((statusCount, 1))

        for words in trainingList:
            statusInit = 0
            for index, wordItems in enumerate(words):
                if index > 0:
                    break

                if len(wordItems) == 1:
                    statusInit = 1
                else:
                    statusInit = 0

            piExpect[statusInit, 0] += 1
        
        piExpect = piExpect / np.sum(piExpect)
        return piExpect

    def getAExpected(self, trainingList):
        statusCount = len(STATUS)
        AExpect = np.zeros((statusCount, statusCount))

        for words in trainingList:
            statusList = list()
            for wordItems in words:
                if len(wordItems) == 1:
                    statusList.append(1)
                else:
                    statusList.append(0)
                    for k in range(1, len(wordItems)):
                        if k < (len(wordItems) - 1):
                            statusList.append(0)
                        else:
                            statusList.append(1)
            
            for index, status in enumerate(statusList):
                if index + 1 < len(statusList):
                    i = status
                    j = statusList[index + 1]
                    AExpect[i, j] += 1
        
        AExpectSum = np.sum(AExpect, axis=1)
        AExpectSum = AExpectSum.reshape(-1, 1)
        AExpect /= AExpectSum

        return AExpect

    def getBExpected(self, trainingList):
        statusCount = len(STATUS)
        OCount = len(self.indexToWord)
        BExpect = np.zeros((statusCount, OCount))

        for words in trainingList:
            statusList = list()
            for wordItems in words:
                if len(wordItems) == 1:
                    statusList.append(1)
                else:
                    statusList.append(0)
                    for k in range(1, len(wordItems)):
                        if k < (len(wordItems) - 1):
                            statusList.append(0)
                        else:
                            statusList.append(1)
            
            sentence = ''.join(words)
            assert len(sentence) == len(statusList)

            for word, status in zip(sentence, statusList):
                wordIndex = self.wordToIndex[word]
                BExpect[status, wordIndex] += 1

        BExpectSum = np.sum(BExpect, axis=1)
        BExpectSum = BExpectSum.reshape(-1, 1)
        BExpect /= BExpectSum

        return BExpect

    def alpha(self, i, t, observations):
        word = observations[t]
        wordIndex = self.wordToIndex[word]

        if t == 0:
            result = self.pi[i] * self.B[i][wordIndex]
            return result

        alphaNext = 0.0
        temp = 0.0
        for j in len(STATUS):
            temp += alpha(j, t - 1, observations) * self.A[j][i]
        alphaNext = temp * self.B[i][wordIndex]

        return alphaNext

    # 前向算法
    def forward(self, observations):
        T = len(observations)
        pO = 0.0
        for j in len(STATUS):
            pO += alpha(j, T, observations)
        return pO

    # 后向算法
    def backward(self):
        pass

    def viterbi(self):
        pass

    def baumWelch(self):
        pass

