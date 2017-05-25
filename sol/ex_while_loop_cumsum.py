tf.reset_default_graph()

# input
x_len = 6
x = tf.constant([7, 0, 42, 1, 13, 4])


def cond(i, acc, y):
    return i < x_len


def body(i, acc, y):
    acc += x[i]
    y = tf.concat([y, tf.expand_dims(acc, axis=0)], axis=0)  # y.shape = (?,)
    i += 1
    return i, acc, y


# initial value for y
y_init = tf.zeros(shape=[0], dtype=x.dtype)

# initial values for the loop variables
loop_vars = [0, 0, y_init]

# specify dynamic shape invariant for y
shape_invariants = [
    tf.TensorShape([]),      # i.shape
    tf.TensorShape([]),      # acc.shape
    tf.TensorShape([None]),  # y.shape
]

# compute the loop
i, acc, y = tf.while_loop(cond, body, loop_vars, shape_invariants)


with tf.Session() as s:
    print s.run(tf.cumsum(x))
    print s.run(y)