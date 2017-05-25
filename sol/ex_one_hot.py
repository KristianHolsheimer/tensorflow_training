# clear any previous computation graphs
tf.reset_default_graph()

# dimensions
n_chars = 129
batch_size = 2
max_seqlen = None  # varies with each batch

# input placeholder
sents_enc = tf.placeholder(shape=[batch_size, max_seqlen], dtype=tf.int32)

# sparse representation
x_one_hot = tf.one_hot(sents_enc, n_chars)

# input
sents = ['Hello, world!', 'Bye bye.']


with tf.Session() as s:
    x_one_hot_ = s.run(x_one_hot, feed_dict={sents_enc: encoder(sents)})
    print x_one_hot_.shape  # (2, 14, 129)