'''
A Gentle Introduction to Graph Neural Networks (Basics, DeepWalk, and GraphSage)
notes from
https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3

with background...
https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
with referenced paper
https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf


Applications:
Node classification (predicting class of node)

each node v has feature x_v, label t_v.

given a partially labeled graph, predict the labels of the unlabeled nodes
it represents each node with a d-dimensional hidden (latent) vector h_v

h_v = f(x_v, xco_v, hne_v, xne_v )

h_v   is embeddings of v
x_v   is features of v
xco_v is features of edges associated w/ v
hne_v is embeddings of nodes associated w/ v
xne_v is features   of nodes associated w/ v

in other words - embeddings results from self features, edge features, neighbor features and embeddings.

Ht+1 = f(Ht, X)
where Ht is concatenation of all embeddings, X concatenation of all features
this is "message passing" or "neighborhood aggregation".

Output function o_v = g(h_v, x_v)

f and g can be considered feed-forward fully connected neural networks.
L1 loss can be calculated as  sum_over_p( ti - oi ) where "p" is the set of labeled nodes (?)

Three limitations:
1) assumption of "fixed point" ?
2) cannot process edge information where different edges represent different things
3) "fixed point" can discourage the diversification of the node distribution

DeepWalk
node embedding learning in unsupervised manner
motivation is that nodes follow power law (log # vertices vs log vertex visitation count) is linear.

Algorithm:
Perform random walks in graph to generate node sequences
Run skip-gram to learn embedding of each node based on sequence above
Each sequence is truncated into sub-sequences of 2(w)+1 where w is window size in skip-gram

<skip-gram at>
https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c

'''

# no code - just notes
