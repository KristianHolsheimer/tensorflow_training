n_hidden = 5
n_output = 3  # can be at most n_hidden


# memory cell *output* (from previous timestep)
h = np.random.random([1, n_hidden])

# candidate memory cell *c-state* (from previous timestep)
c = np.random.random([1, n_hidden])

# memory cell *m-state* (from previous timestep)
m = np.random.random([1, n_hidden])

# input gate weights
w_i = np.random.random([n_features, n_hidden])
u_i = np.random.random([n_hidden, n_hidden])
b_i = np.random.random([1, n_hidden])

# candidate memory (c-state) weights
w_c = np.random.random([n_features, n_hidden])
u_c = np.random.random([n_hidden, n_hidden])
b_c = np.random.random([1, n_hidden])

# forget gate weights
w_f = np.random.random([n_features, n_hidden])
u_f = np.random.random([n_hidden, n_hidden])
b_f = np.random.random([1, n_hidden])

# output gate weights
w_o = np.random.random([n_features, n_hidden])
u_o = np.random.random([n_hidden, n_hidden])
v_o = np.random.random([n_hidden, n_hidden])
b_o = np.random.random([1, n_hidden])

# output projection weights
w_h = np.random.random([n_hidden, n_output])
b_h = np.random.random([1, n_output])
