# %%
""" Primeiro trabalho de IA
    Aluno:  Gabriel Ferrari Cipriano
    Tema:  Algoritmos de Busca e Clusterização
    Trabalho : //TODO
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris, load_wine

from problema.clustering import Clustering
from problema.utils import evaluate_dists_state
from heuristics import grasp, simulated_annealing, genetic

iris = sns.load_dataset("iris")
print(iris)
# output = sns.scatterplot(data=iris, 
#                         x='sepal_length', 
#                         y='petal_width',
#                         hue='species')

iris_data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
data = np.asarray(iris_data)
iris_problem = Clustering(data)

# %%

wine = sns.load_dataset("wine")
print(wine)
# %%
centroids, i, t, labels, min_dists = grasp(iris_problem,
                                              k = 10,
                                              num_best = 5,
                                              max_iter = 60,
                                              max_time = 1.)

sse = evaluate_dists_state(min_dists)

iris_data["cluster"] = labels

output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
                         hue='cluster')

print("k = 10 \n sse:", sse, "\n iter: ", i)


# %%
centroids, t = simulated_annealing(iris_problem, k = 10,
                                                 t = 50,
                                                 alfa = 0.7,
                                                 min_t = 1,
                                                 num_iter = 350,
                                                 tempo = 1)

labels, min_dists = iris_problem.assign_clusters(centroids)

sse = evaluate_dists_state(min_dists)

iris_data["cluster"] = labels

output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
                         hue='cluster')

print("k = 10 \n sse:", sse, "\n tempo: ", t)


# %%
centroids, t, i = genetic(iris_problem, k = 3,   t_pop = 10,
                                                 taxa_cross = 0.75,
                                                 taxa_mutacao = 0.10)

labels, min_dists = iris_problem.assign_clusters(centroids)

sse = evaluate_dists_state(min_dists)

iris_data["cluster"] = labels

output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
                         hue='cluster')

print("k = 3 \n sse:", sse, "\n tempo: ", t, "\ni = ", i)

# %%
