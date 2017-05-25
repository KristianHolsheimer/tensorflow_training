from scipy.special import expit


def init_weights_numpy(shape):
    np.random.seed(sum(shape))
    w = np.random.randn(*shape)
    stddev = 6.0 / np.sqrt(sum(shape))
    np.random.seed(None)
    return stddev * w


def get_n_features(x):
    return x.shape[1]


# defining things like this will make it easier to port to tensorflow
sigmoid = expit
tanh = np.tanh
dot = np.dot
init_weights = init_weights_numpy


class MyLSTMCell:
    def __init__(self, n_hidden, n_output):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_features = None
        
    def _init_weights(self, n_features):
        if self.n_features is not None:
            if self.n_features != n_features:
                raise ValueError("input x doesn't have a fixed number of features")
            return
        
        n_hidden = self.n_hidden
        n_output = self.n_output

        # input gate weights
        self.w_i = init_weights([n_features, n_hidden])
        self.u_i = init_weights([n_hidden, n_hidden])
        self.b_i = init_weights([1, n_hidden])

        # candidate memory (c-state) weights
        self.w_c = init_weights([n_features, n_hidden])
        self.u_c = init_weights([n_hidden, n_hidden])
        self.b_c = init_weights([1, n_hidden])

        # forget gate weights
        self.w_f = init_weights([n_features, n_hidden])
        self.u_f = init_weights([n_hidden, n_hidden])
        self.b_f = init_weights([1, n_hidden])

        # output gate weights
        self.w_o = init_weights([n_features, n_hidden])
        self.u_o = init_weights([n_hidden, n_hidden])
        self.v_o = init_weights([n_hidden, n_hidden])
        self.b_o = init_weights([1, n_hidden])
        
        # output gate weights
        self.w_h = init_weights([n_hidden, n_output])
        self.b_h = init_weights([1, n_output])
        
        # set n_features
        self.n_features = n_features
        
    def __call__(self, x, state):
        self._init_weights(get_n_features(x))
        
        # unpack state
        c, m = state

        # input gate
        i = sigmoid(dot(x, self.w_i) + self.b_i)

        # forget gate
        f = sigmoid(dot(x, self.w_f) + dot(m, self.u_f) + self.b_f)

        # candidate memory (c-state)
        c = f * c + i * tanh(dot(x, self.w_c) + dot(m, self.u_c) + self.b_c)

        # output gate
        o = sigmoid(dot(x, self.w_o) + dot(m, self.u_o) + dot(c, self.v_o) + self.b_o)

        # memory cell state (m-state)
        m = o * tanh(m)

        # output
        h = tanh(dot(m, self.w_h) + self.b_h)
        
        # pack state as a tuple
        state = (c, m)

        return h, state



x = np.random.random([1, n_features])
c = init_weights([1, n_hidden])
m = init_weights([1, n_hidden])

lstm_cell = MyLSTMCell(n_hidden, n_output)

h_numpy, _ = lstm_cell(x, (c, m))
h_numpy