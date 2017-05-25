# input sentences
sents = ['Hello, world!', 'Bye bye.']

# this is the expected output
out = [[ 73, 102, 109, 109, 112,  45,  33, 120, 112, 115, 109, 101,  34,   0],
       [ 67, 122, 102,  33,  99, 122, 102,  47,   0,   0,   0,   0,   0,   0]]


def encoder(sents):
    # do the encoding
    sents_enc = [[ord(char) + 1 for char in sent] for sent in sents]
    
    # get the dimensions
    seqlen = map(len, sents_enc)
    max_seqlen = max(seqlen)
    batch_size = len(sents)
    
    # right-pad with zeros
    sents_enc = [
        s + [0] * (max_seqlen - l + 1)
        for s, l in zip(sents_enc, seqlen)]
    
    return sents_enc
    

print encoder(sents)
np.testing.assert_array_equal(out, encoder(sents))