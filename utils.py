from collections import Iterable
import math
import pandas as pd
import numpy as np


class SentenceEncoder:
    """
    Create a generator that generates batches of sentences of size ``batch_size``.

    N.B. The EOS character is encoded as 0.

    Params
    ------
    seq : seq of str
        Sequence of strings to be encoded.

    Yields
    ------
    sents_unencoded, sents_encoded, seqlen
        Here, ``sents_unencoded`` is a sequence of the unencoded sentences,
        ``sents_unencoded`` is a sequence of the
    """
    def __init__(self, seq, batch_size=128):
        if not isinstance(seq, Iterable) or isinstance(seq, basestring):
            raise TypeError("`sents` must be a sequence of strings")
        self.seq = pd.Series(seq)
        self.batch_size = batch_size

    def __iter__(self):
        self._seq = self.seq.sample(frac=1).reset_index(drop=True)
        self._n_batches = int(math.ceil(self._seq.size / float(self.batch_size)))
        self._seq.index = self._seq.index % self._n_batches
        self._counter = 0
        return self

    def next(self):
        if self._counter >= self._n_batches:
            raise StopIteration

        batch = self._seq[self._counter]
        self._counter += 1
        if isinstance(batch, basestring):
            batch = np.array([batch])
        if batch.shape[0] < self.batch_size:
            batch = np.concatenate([batch, self._seq.sample(self.batch_size - batch.shape[0]).values])
        else:
            batch = batch.values
        return self._encode(batch)

    def _encode(self, seq):
        # encode at the character level
        seq_encoded_variable_shape = [
            [ord(t) + 1 for t in s.encode('ascii', 'replace')] + [0] for s in seq]
        seqlen = np.array(map(len, seq_encoded_variable_shape), dtype=np.int32)
        max_seqlen = max(seqlen)

        # create zero-padded array of encoded sentences
        seq_encoded = np.zeros([len(seq), max_seqlen], dtype=np.int32)
        seq_mask = np.zeros([len(seq), max_seqlen], dtype=np.bool)
        for i, (token_ids, length) in enumerate(zip(seq_encoded_variable_shape, seqlen)):
            seq_encoded[i, :length] = token_ids
            seq_mask[i, :length] = 1

        return seq_encoded, seqlen, seq_mask, max_seqlen

    def decode(self, seq_encoded):
        seq_decoded = []
        for s in seq_encoded:
            string = ""
            for t in s:
                if t == 0:
                    break
                string += chr(t - 1)
            seq_decoded.append(string)
        return seq_decoded
