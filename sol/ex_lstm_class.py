from scipy.special import expit as sigmoid


class MyLSTMCell:
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        self.n_features = None
        
    def _init_weights(self, n_features):
        if self.n_features is not None:
            if self.n_features != n_features:
                raise ValueError("input x doesn't have a fixed number of features")
            return

        n_hidden = self.n_hidden

        # memory cell *output* (from previous step)
        self.h = np.random.random([1, n_hidden])

        # memory cell *state* (from previous step)
        self.m = np.random.random([1, n_hidden])

        # input gate
        self.w_i = np.random.random([n_features, n_hidden])
        self.u_i = np.random.random([n_hidden, n_hidden])
        self.b_i = np.random.random([1, n_hidden])

        # candidate memory cell weights
        self.w_c = np.random.random([n_features, n_hidden])
        self.u_c = np.random.random([n_hidden, n_hidden])
        self.b_c = np.random.random([1, n_hidden])

        # forget gate weights
        self.w_f = np.random.random([n_features, n_hidden])
        self.u_f = np.random.random([n_hidden, n_hidden])
        self.b_f = np.random.random([1, n_hidden])

        # output gate weights
        self.w_o = np.random.random([n_features, n_hidden])
        self.u_o = np.random.random([n_hidden, n_hidden])
        self.v_o = np.random.random([n_hidden, n_hidden])
        self.b_o = np.random.random([1, n_hidden])
        
        # set n_features
        self.n_features = n_features
        
    def __call__(self, x):
        self._init_weights(x.shape[1])

        # input gate
        i = sigmoid(np.dot(x, self.w_i) + self.b_i)

        # candidate memory
        c = np.tanh(np.dot(x, self.w_c) + np.dot(self.h, self.u_c) + self.b_c)

        # forget gate
        f = sigmoid(np.dot(x, self.w_f) + np.dot(self.h, self.u_f) + self.b_f)

        # memory cell state
        self.m = i * c + f * self.m

        # output gate
        o = sigmoid(np.dot(x, self.w_o) + np.dot(self.h, self.u_o) + np.dot(self.m, self.v_o) + self.b_o)

        # output
        self.h = o * np.tanh(self.m)
        
        return self.h


lstm_cell = MyLSTMCell(5)
lstm_cell(x)