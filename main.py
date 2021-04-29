# # %%
# """ Primeiro trabalho de IA
#     Aluno:  Gabriel Ferrari Cipriano
#     Tema:  Algoritmos de Busca e Clusterização
#     Trabalho : //TODO
# """

# import numpy as np
# import seaborn as sns

# from sklearn.datasets import load_iris, load_wine

# from problema.clustering import Clustering
# from problema.utils import evaluate_dists_state
# from heuristics import grasp, simulated_annealing, genetic

# iris = sns.load_dataset("iris")
# print(iris)
# # output = sns.scatterplot(data=iris, 
# #                         x='sepal_length', 
# #                         y='petal_width',
# #                         hue='species')

# iris_data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# data = np.asarray(iris_data)
# iris_problem = Clustering(data)

# # %%

# wine = sns.load_dataset("wine")
# print(wine)
# # %%
# centroids, i, t, labels, min_dists = grasp(iris_problem,
#                                               k = 10,
#                                               num_best = 5,
#                                               max_iter = 60,
#                                               max_time = 1.)

# sse = evaluate_dists_state(min_dists)

# iris_data["cluster"] = labels

# output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
#                          hue='cluster')

# print("k = 10 \n sse:", sse, "\n iter: ", i)


# # %%
# centroids, t = simulated_annealing(iris_problem, k = 10,
#                                                  t = 50,
#                                                  alfa = 0.7,
#                                                  min_t = 1,
#                                                  num_iter = 350,
#                                                  tempo = 1)

# labels, min_dists = iris_problem.assign_clusters(centroids)

# sse = evaluate_dists_state(min_dists)

# iris_data["cluster"] = labels

# output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
#                          hue='cluster')

# print("k = 10 \n sse:", sse, "\n tempo: ", t)


# # %%
# centroids, t, i = genetic(iris_problem, k = 3,   t_pop = 10,
#                                                  taxa_cross = 0.75,
#                                                  taxa_mutacao = 0.10)

# labels, min_dists = iris_problem.assign_clusters(centroids)

# sse = evaluate_dists_state(min_dists)

# iris_data["cluster"] = labels

# output = sns.scatterplot(data=iris_data, x='sepal_length', y='petal_width',
#                          hue='cluster')

# print("k = 3 \n sse:", sse, "\n tempo: ", t, "\ni = ", i)

# %%

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from problema.clustering import Clustering
from problema.utils import *
from heuristics import grasp, simulated_annealing, genetic

from sklearn.datasets import load_iris, load_wine
from pandas import read_csv

from problema.training import TrainGrasp, TrainSA, TrainGenetic
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    # TRAIN
    iris = load_iris()['data']
    wine = load_wine()['data']
    print(iris.shape)
    print(wine.shape)

    iris_ks = [3, 7, 10, 13, 22]
    wine_ks = [2, 6, 9, 11, 33]


    grasp_params = [[20, 50, 100, 200, 350, 500], [5, 10, 15]]
    sa_params    = [[500., 100., 50.], [0.95, 0.85, 0.7], [350, 500]]
    gen_params   = [[10, 30, 50], [0.75, 0.85, 0.95], [0.10, 0.20]]

    iris_problem = Clustering(iris)
    wine_problem = Clustering(wine)

    grasp_iris = TrainGrasp(iris_problem, grasp_params, iris_ks)
    grasp_wine = TrainGrasp(wine_problem, grasp_params, wine_ks)

    sa_iris = TrainSA(iris_problem, sa_params, iris_ks)
    sa_wine = TrainSA(wine_problem, sa_params, wine_ks)

    gen_iris = TrainGenetic(iris_problem, gen_params, iris_ks)
    gen_wine = TrainGenetic(iris_problem, gen_params, wine_ks)

    results = {"sa": {}, "grasp": {}, "gen": {}}
    hparams = {
                "sa": sa_iris.hparams,
                "grasp" : grasp_iris.hparams,
                "gen" : gen_iris.hparams
                }

    print("Training started")
    grasp_iris.train(10)
    grasp_wine.train(10)
    results["grasp"]["iris"] = grasp_iris.result
    results["grasp"]["wine"] = grasp_wine.result
    print("GRASP trained.")

    sa_iris.train(10)
    sa_wine.train(10)
    results["sa"]["iris"] = sa_iris.result
    results["sa"]["wine"] = sa_wine.result
    print("SA trained.")

    gen_iris.train(10)
    gen_wine.train(10)
    results["gen"]["iris"] = gen_iris.result
    results["gen"]["wine"] = gen_wine.result
    print("GA trained.")

    with open('train_results_11april.json', 'w', encoding='utf-8') as outfile:
        json.dump({"results":results, "hparams":hparams}, outfile, cls=NumpyEncoder, indent=2)

if __name__=="__main__":
    main()