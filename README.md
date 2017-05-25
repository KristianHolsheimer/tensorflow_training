# Introduction to Recurrent Neural Networks in Tensorflow

In this tutorial you'll learn how to create a recurrent neural network that predicts the next character from a short sequence of *seed* characters, e.g.
```python
In [1]: seed = "We always "
In [2]: model.finish_text(seed)
Out[2]: 'We always finish each others sentences.'
```
The aim of this course is to build intuition for how the internals work. This means that some fancy techniques are left out to avoid clutter in the presentation.

## Predicting the next character in a string using LSTMs

We split up our approach into the following smaller easy-to-follow notebooks:

### [Notebook 1](intro_general.ipynb): Some general background about Tensorflow
### [Notebook 2](build_your_own_lstm_cell.ipynb): Build your own Long Short-Term Memory (LSTM) cell.
### [Notebook 3](control_flow_in_tensorflow.ipynb): Control flow in Tensorflow (conditionals and loops).
### [Notebook 4](text_data_representation.ipynb): Sparse and dense representations of text data.
### [Notebook 5](build_your_own_rnn.ipynb): Build your own Recurrent Neural Network (RNN).



***Note:*** *This is still a work in progress. Feel free to comment, or better yet, contribute.*
