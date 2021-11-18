import numpy as np

STATUS = {0, 1}   # 状态集合，0表示非结尾字，1表示结尾字

class Hmm:
    def __init__(self, A, B, pi, word2Index, index2Word):
        self.A = A      # A为状态转移概率矩阵
        self.B = B      # B为观测生成概率矩阵
        self.pi = pi    # pi为初始状态矩阵

        self.wordToIndex = word2Index      # 字典库
        self.indexToWord = index2Word
    
    # 使用监督学习计算初始状态矩阵
    @staticmethod
    def getPiExpected(trainingList):
        statusCount = len(STATUS)
        piExpect = np.zeros(statusCount)

        for words in trainingList:
            statusInit = 0
            for index, wordItems in enumerate(words):
                if index > 0:
                    break

                if len(wordItems) == 1:
                    statusInit = 1
                else:
                    statusInit = 0

            piExpect[statusInit] += 1
        
        piExpect = piExpect / np.sum(piExpect)
        return piExpect

    # 使用监督学习计算观测生成概率矩阵
    @staticmethod
    def getAExpected(trainingList):
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

    # 使用监督学习计算状态转移概率矩阵
    @staticmethod
    def getBExpected(trainingList, indexToWord, wordToIndex):
        statusCount = len(STATUS)
        OCount = len(indexToWord)
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
                wordIndex = wordToIndex[word]
                BExpect[status, wordIndex] += 1

        BExpectSum = np.sum(BExpect, axis=1)
        BExpectSum = BExpectSum.reshape(-1, 1)
        BExpect /= BExpectSum

        return BExpect

    def alpha(self, i, t, observations):
        # print('t = ', t)

        word = observations[t]
        wordIndex = self.wordToIndex[word]

        if t == 0:
            result = self.pi[i] * self.B[i][wordIndex]
            return result

        alphaNext = 0.0
        for j in range(len(STATUS)):
            alphaNext += self.alpha(j, t - 1, observations) * self.A[j][i]
        alphaNext *= self.B[i][wordIndex]

        return alphaNext

    def alphav2(self, observations):
        N = len(STATUS)
        T = len(observations)
        alp = np.zeros((N, T))

        word0 = observations[0]
        wordIndex0 = self.wordToIndex[word0]
        alp[:, 0] = self.pi * self.B[:, wordIndex0]
        for t in range(1, T):
            wordt = observations[t]
            wordIndext = self.wordToIndex[wordt]
            for i in range(N):
                alp[i, t] = np.dot(alp[:, t - 1], self.A[:, i]) * self.B[i, wordIndext]

        return alp

    def beta(self, i, t, observations):
        T = len(observations) - 1
        if t == T:
            return 1

        word = observations[t + 1]
        wordIndex = self.wordToIndex[word]
        betaPrev = 0.0
        for j in range(len(STATUS)):
            betaPrev += self.A[i][j] * self.B[j][wordIndex] * self.beta(j, t + 1, observations)
        
        return betaPrev

    def betav2(self, observations):
        N = len(STATUS)
        T = len(observations)
        bet = np.zeros((N, T))

        bet[:, -1] = 1
        for t in reversed(range(T - 1)):
            for i in range(N):
                word = observations[t + 1]
                wordIndex = self.wordToIndex[word]
                bet[i, t] = np.sum(self.A[i, :] * self.B[:, wordIndex] * bet[:, t + 1])

        return bet

    # 前向算法
    def forward(self, observations):
        T = len(observations) - 1
        # print(T)
        pO = 0.0
        for i in range(len(STATUS)):
            pO += self.alpha(i, T, observations)
        return pO

    def forwardv2(self, observations):
        T = len(observations)
        alp = self.alphav2(observations)
        return np.sum(alp[:, T - 1])

    # 后向算法
    def backward(self, observations):
        pO = 0.0
        word = observations[0]
        wordIndex = self.wordToIndex[word]
        for i in range(len(STATUS)):
            pO += self.pi[i] * self.B[i][wordIndex] * self.beta(i, 0, observations)

        return pO

    def backwardv2(self, observations):
        bet = self.betav2(observations)
        word0 = observations[0]
        wordIndex0 = self.wordToIndex[word0]
        return np.sum(self.pi * self.B[:, wordIndex0] * bet[:, 0])
        
    # 直接暴力计算序列生成概率
    def forceCalP(self, observations):
        pass   # hoho_todo


    # 维特比算法
    def viterbi(self):
        pass


    # 使用Baum-Welch训练迭代
    def fit(self, trainingList, e=1e-10):
        for index, words in enumerate(trainingList):
            observations = ''.join(words)
            self.baumWelch(observations)


    def baumWelch(self, O, e=1E-10):
        row=self.A.shape[0]
        col=len(O)
        
        done=False
        epoch = 0
        while not done:
            zeta=np.zeros((row,row,col-1))
            alpha=self.alphav2(O)
            beta=self.betav2(O)
            #EM算法：由 E-步骤 和 M-步骤 组成
            #E-步骤：计算期望值zeta和gamma

            for t in range(col-1):
                #分母部分
                word = O[t + 1]
                wordIndex = self.wordToIndex[word]
                denominator=np.dot(np.dot(alpha[:,t],self.A)*self.B[:,wordIndex],beta[:,t+1])
                for i in range(row):
                    #分子部分以及zeta的值
                    numerator=alpha[i,t]*self.A[i,:]*self.B[:,wordIndex]*beta[:,t+1]
                    zeta[i,:,t]=numerator/denominator
                    
            gamma=np.sum(zeta,axis=1)
            final_numerator=(alpha[:,col-1]*beta[:,col-1]).reshape(-1,1)
            final=final_numerator/np.sum(final_numerator)
            gamma=np.hstack((gamma,final))
            #M-步骤：重新估计参数Pi,A,B
            newPi=gamma[:,0]
            newA=np.sum(zeta,axis=2)/np.sum(gamma[:,:-1],axis=1)
            newB=np.copy(self.B)
            b_denominator=np.sum(gamma,axis=1)

            temp_matrix=np.zeros((1,len(O)))
            for k in range(self.B.shape[1]):
                for t in range(len(O)):
                    wordT = O[t]
                    wordIndexT = self.wordToIndex[wordT]
                    if wordIndexT==k:
                        temp_matrix[0][t]=1
                newB[:,k]=np.sum(gamma*temp_matrix,axis=1)/b_denominator

            pOld = self.forwardv2(O)
            self.A=newA
            self.B=newB
            self.pi=newPi
            # print('new pi: ', self.pi)
            # print('new A: ', self.A)
            # print('new B: ', self.B)
            pNew = self.forwardv2(O)

            # print('pNew: ', pNew * 1E10)
            print('diff: ', (pNew - pOld) * 1E15)
            
            epoch += 1
            if epoch == 1000:
                done = True