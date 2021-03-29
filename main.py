""" Primeiro trabalho de IA
    Aluno:  Gabriel Ferrari Cipriano
    Tema:  Algoritmos de Busca e Clusterização
    Trabalho : //TODO
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from problema.clustering import Clustering
from problema.utils import *
from heuristics import grasp, simulated_annealing

def main():
    iris = sns.load_dataset("iris")
    print(iris)
    output = sns.scatterplot(data=iris, x='sepal_length', 
                                        y='petal_width',
                                        hue='species')
    plt.show()









main()