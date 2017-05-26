# Introduction to Recurrent Neural Networks in Tensorflow

In this tutorial you'll learn how to create a recurrent neural network that predicts the next character from a short sequence of *seed* characters, e.g.
```python
In [1]: seed = "We always "
In [2]: model.finish_text(seed)
Out[2]: 'We always finish each others sentences.'
```
The aim of this course is to build intuition for how the internals work. This means that some fancy techniques are left out to avoid clutter in the presentation.

Solutions to all the exercises are provided inline like so:
```python
# %load sol/ex_foo.py
```
The way to load the exercise is to uncomment the line and run the `%load` magic (see [docs](https://ipython.org/ipython-doc/3/interactive/magics.html#magic-load)). This will load the solution into the cell where the `%load ...` was evaluated. So, these `%load`s allow you to peek at the correct solution. I advice not to turn to the solutions to quickly. Give yourself a few minutes to think about how you would approach the problem. That way, your mind is primed to soak up every bit of information provided in the solution.

## Predicting the next character in a string using LSTMs

We split up our approach into the following smaller easy-to-follow notebooks:

### [Notebook 1](intro_general.ipynb): Some general background about Tensorflow
### [Notebook 2](build_your_own_lstm_cell.ipynb): Build your own Long Short-Term Memory (LSTM) cell.
### [Notebook 3](control_flow_in_tensorflow.ipynb): Control flow in Tensorflow (conditionals and loops).
### [Notebook 4](text_data_representation.ipynb): Sparse and dense representations of text data.
### [Notebook 5](build_your_own_rnn.ipynb): Build your own Recurrent Neural Network (RNN).



***Note:*** *This is still a work in progress. Feel free to comment, or better yet, contribute.*
