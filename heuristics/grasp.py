"""
    Grasp metaheuristic for clustering problem
"""
import time
import random as rand
import numpy as np

from problema.clustering import Clustering, escolhe_melhores, evaluate_state

def construcao_gulosa(problem: Clustering, state, num_best, max_time):
    '''
        construção gulosa do GRASP
    '''
    start = time.process_time()
    end = 0.

    k = len(state)

    centroids = state.copy()

    while end-start <= max_time:
        aux_centroids = centroids.copy()
        labels, min_dists = problem.assign_clusters(aux_centroids)

        nbhood = problem.generate_nbhood(aux_centroids, labels, min_dists)
        nbhood = escolhe_melhores(nbhood, num_best)

        # Pegando index e label do estado vizinho aleatoriamente escolhido
        obs_index, label, _ = nbhood[rand.randint(0,len(nbhood)-1)]

        # Atribuindo novo estado
        labels[obs_index] = label

        # Calculando novo estado (centroides) da nova label
        centroids, has_members = problem.update_centroids(labels, k)

        # Caso algum centroide não possua membros, atribui a posicao anterior
        if not has_members.all():
            # Setting to previous centroid position
            centroids[~has_members] = aux_centroids[~has_members]

        end = time.process_time()

    return centroids

# local searh based on kmeans algorithn
def local_search(problem: Clustering, state, max_time):
    '''
        Busca local baseada no k-means
    '''
    start = time.process_time()
    end = 0.

    k = len(state)
    centroids = state.copy()

    # main loop
    # for i in range(iter):
    while end-start <= max_time:
        labels, _ = problem.assign_clusters(centroids)

        # update centroids
        new_centroids, has_members = problem.update_centroids(labels, k)

        # check convergence
        if np.array_equal(new_centroids, centroids):
            break

        if not has_members.all():
            # Setting to previous position
            new_centroids[~has_members] = centroids[~has_members]

        centroids = new_centroids
        end = time.process_time()

    return centroids

def grasp(problem: Clustering, k, num_best = 5, max_iter = 20, max_time = 1.):
    '''
        Grasp for clustering problems
    '''
    start = time.process_time()
    end = 0
    count = 0

    opt_centroids = problem.generate_initial_centroids(k)
    opt_labels, opt_min_dists = problem.assign_clusters(opt_centroids)
    opt_sse = evaluate_state(opt_min_dists)

    while (end-start) < max_time and count < max_iter:
        # Construção
        new_centroids = problem.generate_initial_centroids(k)
        new_centroids = construcao_gulosa(problem, new_centroids, num_best, max_time/max_iter)
        # Local search
        new_centroids = local_search(problem, new_centroids, max_time/10)

        new_labels, new_min_dists = problem.assign_clusters(new_centroids)
        new_sse = evaluate_state(new_min_dists)

        if new_sse < opt_sse:
            opt_centroids = new_centroids
            opt_labels = new_labels
            opt_min_dists = new_min_dists
            opt_sse = new_sse

        end = time.process_time()
        count += 1

    return opt_centroids, count, opt_labels, opt_min_dists
