
# clear previous computation graph
tf.reset_default_graph()

# some test casesx = tf.constant(7)
y = tf.constant(-13)


def my_abs(x):
    return tf.cond(x >= 0, lambda: x, lambda: -x)


with tf.Session() as s:
    print "%3s => %3s" % s.run((x, my_abs(x)))
    print "%3s => %3s" % s.run((y, my_abs(y)))