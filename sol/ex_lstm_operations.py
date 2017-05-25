from scipy.special import expit as sigmoid

# input gate
i = sigmoid(np.dot(x, w_i) + b_i)

# forget gate
f = sigmoid(np.dot(x, w_f) + np.dot(m, u_f) + b_f)

# candidate memory (c-state)
c = f * c + i * np.tanh(np.dot(x, w_c) + np.dot(m, u_c) + b_c)

# output gate
o = sigmoid(np.dot(x, w_o) + np.dot(m, u_o) + np.dot(c, v_o) + b_o)

# memory cell state (m-state)
m = o * np.tanh(m)

# output
h = np.tanh(np.dot(m, w_h) + b_h)

assert h.shape == (1, n_output)
h