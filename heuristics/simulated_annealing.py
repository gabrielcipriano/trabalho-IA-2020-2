""" Simulated Annealing
        Tipo: Baseada em Soluções Completas -> Busca Local
        Precisa de Estado Inicial : Nao
        Hiperparametros:
            t: temperatura inicial
            alfa: taxa de queda de temperatura
            min_t: temperatura mínima (critério de parada)
            num_iter: numero de iterações por temperatura
"""
import random as rand
import time

from problema.clustering import Clustering
from problema.utils import generate_labels_nbhood

def simulated_annealing(problema: Clustering, k, t, alfa, num_iter, min_t, tempo = 1.):
    """Pertuba o estado atribuindo uma label diferente à um ponto aleatório
    """
    start = time.process_time()
    end = 0
    num_obs = len(problema.data)

    opt_state = problema.generate_initial_centroids(k)

    aux_state = opt_state.copy()

    while t > min_t and (end-start) < tempo:
        aux_labels = problema.assign_clusters(aux_state) [0]
        nbhood = generate_labels_nbhood(aux_labels, k)
        for _ in range(num_iter):
            if (end-start) > tempo:
                break
            nb_labels = aux_labels.copy()
            rand_index = rand.randrange(num_obs)
            nb_labels[rand_index] = nbhood[rand_index]

            # Retorna se o vizinho foi aceito, e o estado dele
            nb_accepted, nb_state = problema.accept_neighbour(aux_state, nb_labels, k, t)

            if nb_accepted:
                aux_state = nb_state.copy()
                aux_labels = problema.assign_clusters(aux_state)[0]
                nbhood = generate_labels_nbhood(aux_labels, k)

                if (problema.best_state([aux_state, opt_state]) == aux_state).all():
                    opt_state = aux_state.copy()

            end = time.process_time()

        t = t*alfa

    return opt_state, end-start
