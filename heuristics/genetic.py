""" Metodo Algoritmo Genético
        Hiperparametros : 
            max_iter : número máximo de iterações (critério de parada)
            t_pop : tamanho da população
            taxa_cross : chance de ocorrer crossover
            taxa_mutacao : chance de ocorrer mutação

"""
import time
import random as rand
import numpy as np
from problema.clustering import Clustering


def crossover(state1, state2):
    num_clusters = len(state1)
    # Quantidade aleatoria de crossovers
    qtd = rand.randint(0, num_clusters)

    for _ in range(qtd):
        cromossomo = rand.randrange(num_clusters)
        aux = state1[cromossomo]
        state1[cromossomo] = state2[cromossomo]
        state2[cromossomo] = aux


def genetic(problem: Clustering, k, t_pop, taxa_cross, taxa_mutacao, t = 1., max_sem_melhora = 30, max_iter = 3000):
    """ Parametros : 
            problem : uma instancia do problema de clustering
            k : quantidade de centroides
            t_pop : tamanho da população
            taxa_cross : chance de ocorrer crossover
            taxa_mutacao : chance de ocorrer mutação
            max_sem_melhora : quantidade maxima de iteracoes sem melhora (critério de parada)
            max_iter : número máximo de iterações (critério de parada)
            t : tempo
    """
    start = time.process_time()
    end = 0

    if k == 1:
        return [problem.get_centroid()], time.process_time()-start, 1

    melhor = problem.generate_initial_centroids(k)
    populacao = [melhor]

    melhor_sse = np.inf
    qtd_geracoes_sem_melhora = 0

    i = 0

    while i < max_iter and qtd_geracoes_sem_melhora < max_sem_melhora and end-start < t:
        # Seleciona um estado com potencial e gera a população
        problem.selecao(populacao)
        problem.gerar_populacao(populacao, t_pop, k)

        # Realiza um numero aleatorio de crossovers e mutacoes, 
        # entre 1 e metade do tamanho da populacao
        for _ in range(1, t_pop//2):
            # Crossover
            if taxa_cross >= rand.random():
                x = rand.randrange(len(populacao))
                y = rand.randrange(len(populacao))
                # while x == y:
                #     y = rand.randrange(len(populacao))
                if x == y:
                    y = y-1
                crossover(populacao[x], populacao[y])
            # Mutacao
            if taxa_mutacao >= rand.random():
                x = rand.randrange(len(populacao))
                populacao[x] = problem.mutacao(populacao[x])
    
            # end = time.process_time()

        melhor_da_geracao, melhor_sse_geracao = problem.melhor_da_geracao(populacao)

        if melhor_sse_geracao < melhor_sse:
            melhor = melhor_da_geracao
            melhor_sse = melhor_sse_geracao
            qtd_geracoes_sem_melhora = 0
        else:
            qtd_geracoes_sem_melhora += 1

        i += 1
        end = time.process_time()

    return melhor, end-start, i
