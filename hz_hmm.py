STATUS = {'M', 'E'}   # 状态集合，M表示不是结尾字，E表示结尾字

class HzHmm:
    def __init__(self, A, B, pi, word_to_index, index_to_word):
        self.A = A      # A为状态转移概率矩阵
        self.B = B      # B为观察生成概率矩阵
        self.pi = pi    # pi为初始状态矩阵
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def alpha_i(self, status_i, t, observation_list):
        word = observation_list[t]
        word_index = self.word_to_index[word]

        if t == 0:
            result = self.pi[status_i] * self.B[status_i][word_index]
            return result

        result = 0.0
        temp = 0.0
        for j in len(STATUS):
            temp += alpha_i(j, t - 1, observation_list) * self.A[j][status_i]
        result = temp * self.B[status_i][word_index]
        return result

    def forward(self, observation_list):
        T = len(observation_list)
        temp_i = 0.0
        for j in len(STATUS):
            temp_i = alp
        pass

    def backward(self):
        pass

    def viterbi(self):
        pass

    def baumWelch(self):
        pass

