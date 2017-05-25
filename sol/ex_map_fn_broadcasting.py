tf.reset_default_graph()


a = tf.ones([2, 3, 5])
b = tf.ones([7, 5])

    
c = tf.map_fn(lambda x: a - x, b)
print c.get_shape()