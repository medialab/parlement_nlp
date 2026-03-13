"""
Simple BERT+MLP model detailed in « ParlVote : A Corpus for Sentimental Analysis of Political Debates » and then in « GPolS : A Contextual Graph-Based Language Model for Analyzing Parliamentary Debates and Political Cohesion ».

Architecture detailed as followed :

Multi-layer perceptron (MLP) A simple 'vanilla' neural network, which has been shown to perfom better than SVMs in some circumstances on this task (Abercrombie and Batista-Navarro, 2018a). We used a network with one hidden layer comprised of 100 nodes, batch normalisation, a ReLu activation function, a dropout regularization rate of 0.5, and sigmoid activation in the output layer. We used early stopping with a tolerance of three epochs to select the model used for classification of the examples in the test set.

(...)

We use a MLP with 1 hidden layer containing 100 units followed by ReLU activation. We use L-BFGS optimisation (Liu and Nocedal, 1989) for training 200 epochs (...) BERT (Devlin et al., 2019) embeddings are used on the ParlVote dataset followed by a MLP with the same settings as described above.

"""
import torch
from torch import nn

class ParlMLP(nn.Module):
    def __init__(self, embedding_dim = 768):
        super.__init__()

        self.stack = nn.Sequential(
            nn.Linear(embedding_dim, 100),
            nn.BatchNorm1d(100),
            
        )