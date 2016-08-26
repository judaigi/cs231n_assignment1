import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # for softmax gradient
  # http://mccormickml.com/2014/06/13/deep-learning-tutorial-softmax-regression/
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # for numerial stability http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(scores)
    scores -= log_c

    normalization = 0.0
    for j in xrange(num_classes):
      normalization += np.exp(scores[j])
    loss += - scores[y[i]] 
    loss += np.log(normalization)  

    for j in xrange(num_classes):
      prob = np.exp(scores[j]) / normalization
      dW[:,j] += prob * X[i,:]
      dW[:,j] -= (j==y[i]) * X[i,:]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.matrix(np.max(scores,axis=1)).T
  normalization = np.exp(scores).sum(axis=1)
  loss = - scores[xrange(num_train), y].sum() 
  loss += np.log(normalization).sum() 

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  prob = np.exp(scores) / np.matrix(normalization).T
  dW += X.T.dot(prob)
  match = np.zeros(scores.shape)
  match[xrange(num_train), y] = -1
  dW += X.T.dot(match)

  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

