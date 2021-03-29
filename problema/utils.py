'''
    Util functions to the clustering problem
'''
import random as rand
import numpy as np


def evaluate_state(min_dists):
    '''
        Return: SSE baseado na array de distancias
    '''
    return np.sum(min_dists)

def get_diff_obs_state(current_label, k):
    '''
        Retorna uma label diferente da label atual entre as k disponiveis
    '''
    new_label = rand.randint(0, k-1)
    while new_label == current_label:
        new_label = rand.randint(0, k-1)
    return new_label

def escolhe_melhores(neighbourhood, num_best):
    '''
        Retorna os n melhores de uma vizinhança
    '''
    neighbourhood = np.sort(neighbourhood, order='sse')
    return neighbourhood[:num_best]

def generate_labels_nbhood(labels, k):
    '''
        para cada label dos pontos gera uma label diferente escolhida de maneira aleatoria
    '''
    nbhood = np.empty(len(labels), dtype=int)
    for i, label in enumerate(labels):
        # Modifica o estado (label) do ponto i
        nbhood[i] = get_diff_obs_state(label, k)
    return nbhood
