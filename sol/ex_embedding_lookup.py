# clear any previous computation graphs
tf.reset_default_graph()

# dimensions
n_chars = 129
batch_size = 2
emb_dim = 10
max_seqlen = None  # varies with each batch

# input placeholder
sents_enc = tf.placeholder(shape=[batch_size, max_seqlen], dtype=tf.int32)

# character embeddings
emb = tf.Variable(tf.random_uniform([n_chars, emb_dim]))

# dense representation
x_dense = tf.nn.embedding_lookup(emb, sents_enc)

# input
sents = ['Hello, world!', 'Bye bye.']


with tf.Session() as s:
    s.run(tf.global_variables_initializer())  # init emb
    
    x_dense_ = s.run(x_dense, feed_dict={sents_enc: encoder(sents)})
    print x_dense_.shape  # (2, 14, 10)