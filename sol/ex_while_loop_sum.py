tf.reset_default_graph()

# input
x_len = 6
x = tf.constant([7, 0, 42, 1, 13, 4])


def cond(i, acc):
    return i < x_len


def body(i, acc):
    acc += x[i]
    i += 1
    return i, acc


# initial values for the loop variables
loop_vars = [0, 0]

# compute loop
i, acc = tf.while_loop(cond, body, loop_vars)


with tf.Session() as s:
    print s.run(tf.reduce_sum(x))
    print s.run(acc)