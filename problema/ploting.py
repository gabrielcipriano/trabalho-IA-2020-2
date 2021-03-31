# %%
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sea


def plot_metodo_results(metodo_data, name, hparam_names):
    iris = metodo_data['iris']['results_by_k']
    wine = metodo_data['wine']['results_by_k']
    # Numpy parsing
    for k, _ in iris.items():
        for key, value in iris[k].items():
            iris[k][key] = np.array(value)
    for k, _ in wine.items():
        for key, value in wine[k].items():
            wine[k][key] = np.array(value)

    # # Renaming for label purposes
    # for k in list(iris.keys()):
    #     iris["k"+str(k)+"_iris"] = iris.pop(k)
    # for k in list(wine.keys()):
    #     wine["k"+str(k)+"_wine"] = wine.pop(k)

    df_iris = pd.DataFrame(iris)
    df_wine = pd.DataFrame(wine)

    print(df_iris["3"]["sse_mean"].shape)


    # df_hparams = pd.DataFrame(metodo_data['iris']["hparams"], columns=hparam_names)
    # df_hparams_iris = pd.concat([df_hparams, df_iris], axis=1)
    # df_hparams['mean'] = df_iris.loc['sse_mean'].copy().mean(axis=1)
    # df_hparams['desvio'] = df_iris.loc['sse_mean'].std(axis=0)
    # df_hparams['rank'] = df_iris.loc['rank'].mean(axis=0)

    # print(df_iris.iloc[:,0].shape)
    # print(df_hparams.shape)

    # print(df_hparams_iris)



    '''
        df_iris.iloc[ i, j] :
        i = property index: [sse_mean, t_mean, z_score, rank]
        j = k index: [3,...,22]
    '''
    # plot médias
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel='k', ylabel='Média', title ='Iris')
    ax2.set(xlabel='k', ylabel='Média', title ='wine', ylim =(0, 5000000))
    ax1.grid()
    ax2.grid()
    aux = df_iris.loc["sse_mean"]
    # aux.columns = list(iris.keys())
    sea.boxplot(data=aux, ax=ax1)
    aux = df_wine.loc['sse_mean']
    # # aux.columns = [2, 6, 9, 11, 33]
    sea.boxplot(data=aux, ax=ax2)
    # # plt.tight_layout()
    plt.savefig(name+'_mean.png')

    # boxplot tempos
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel='k', ylabel='Tempo', title ='Iris')
    ax2.set(xlabel='k', ylabel='Tempo', title ='wine')
    ax1.grid()
    ax2.grid()
    aux = df_iris.loc["t_mean"]
    sea.boxplot(data=aux, ax=ax1)
    aux = df_wine.loc["t_mean"]
    sea.boxplot(data=aux, ax=ax2)
    plt.tight_layout()
    plt.savefig(name+'_time.png')



        
# %%
train_data = {}
with open('../train_results.txt') as json_file:
    train_data = json.load(json_file)
# %%
plot_metodo_results(train_data['sa'], "Simulated Annealing", ['t','alfa','iter_max'])
# %%
