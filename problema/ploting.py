# %%
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def get_info_as_df(df, info, configs):
    infos = []
    for problem in df.keys():
        infos.append(df[problem][info].to_numpy())
    infos = pd.DataFrame(np.asarray(infos), 
                        index = df.keys(), 
                        columns= configs)
    return infos

# %%
def build_results_dataframe(data):
    results = data.copy()
    # Renaming for label purposes
    for d in list(results.keys()):
        for p in list(results[d].keys()):
            for k in list(results[d][p].keys()):
                results[d][p+k] = pd.DataFrame(results[d][p][k])
            results[d].pop(p)
    return pd.DataFrame(results, columns = results.keys())

# %%

# %%
def plot_metodo_results(metodo_df, configs, name, hparam_names):
    df = metodo_df
    configs = [tuple([round(i, 2) for i in l]) for l in configs]
    # configs = [tuple(x) for x in configs]

    #  Getting zscore df and tempos df
    zscores = get_info_as_df(df, "zscore", configs)
    tempos = get_info_as_df(df, "t", configs)


    # boxplot zscores and tempos
    figsize = (7.047, 5.022)
    _, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    ax1.set(xlabel='z-score', ylabel='Hiperparâmetros: (' + ', '.join(hparam_names) + ")")
    ax2.set(xlabel='Tempo')
    sns.boxplot(data=zscores, ax=ax1, orient="h", palette="Set3")
    sns.boxplot(data=tempos, ax=ax2,orient="h", palette="Set3")
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(name+'_zscore_and_time_plot.png',dpi=200)
        
# %%
def main():
    train_data = {}
    with open('../train_results_9april.json') as json_file:
        train_data = json.load(json_file)
    results = train_data["results"]
    configs = train_data["hparams"]
    info = {
        "sa": {
            "name": "Simulated Annealing",
            "param_names": ('t','alfa','iter_max')
        },
        "grasp": {
            "name": "GRASP",
            "param_names": ('numIter', 'numBest')
        },
        "gen": {
            "name": "Genetic Algorithm",
            "param_names": ('tamPopulacao', 'tCross', 'tMut')
        },
    }

    df = build_results_dataframe(results)

    metodos = results.keys()
    for m in metodos:
        plot_metodo_results(df[m], configs[m], info[m]["name"], info[m]["param_names"])
# %%
if __name__=="__main__":
    main()
# %%
