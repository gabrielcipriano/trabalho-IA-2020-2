import math as m
import random as rand
import time
import numpy as np


def generate_initial_centroids(data, k):
    # Seleciona aleatoriamente K linhas para serem os centroides
    points = np.random.choice(data.shape[0], size=k, replace=False)
    return data[points]

"""
    Update_cluster_means
        Parameters:
            data : M x N ndarray 
                observation matrix.
            labels : int ndarray
                array of the labels of the observations.
            k : int
                The number of centroids (codes).
        Returns:
            clusters: k x n ndarray
            new centroids matrix
            has_members : ndarray
                A boolean array indicating which clusters have members.
"""
def update_cluster_means(data, labels, k):
    num_obs = len(data)
    num_feat = len(data[0])
    clusters = np.zeros((k, num_feat), dtype=data.dtype)

    # sum of the numbers of obs in each cluster
    obs_count = np.zeros(k, np.int)

    for i in range(num_obs):
        label = labels[i] 
        obs_count[label] += 1
        clusters[label] += data[i]

    for i in range(k):
        cluster_size = obs_count[i]

        if cluster_size > 0:
            # Calculate the centroid of each cluster
            clusters[i] = clusters[i] / cluster_size

    # Return a boolean array indicating which clusters have members
    return clusters, obs_count > 0

from scipy.spatial.distance import cdist
'''
    assign_clusters
    Parametros:
        data: ndarray size M x N
        Cada linha da array é uma observação.
        As colunas são os atributos de cada observação

        centroids: ndarray size k x N
        Cada linha é um centroide
    Retornos:
        labels: ndarray size M
        Uma array contendo o index do cluster atribuido a cada observacao
        min_dist: ndarray size M
        Array contendo a distancia da i-ésima observação até o centroide mais proximo
'''
def assign_clusters(data, centroids):
    dists = cdist(data, centroids, 'sqeuclidean')
    labels = dists.argmin(axis=1)
    min_dist = dists[np.arange(len(labels)), labels]
    
    return labels, min_dist


class Clustering:

    # Descrição da instancia do problema
    def descricao(self):
        pass #TODO

    # Retorna o SSE do estado atual
    def avaliar(self, estado):
        pass

    # Diz se um estado é valido (necessário?) //TODO
    def valido(self, estado):
        pass

    # retorna estado considerado nulo (Existe um estado nulo?) //TODO
    def estadoNulo(self):
        pass

    # Retorna um estado aleatorio (k centroides aleatorios)
    def estadoAleatorio(self):
        pass
    
    # retorna os n melhores estados (centroides) numa lista de estados
    def nMelhores(self, estados, n):
        evaluated_states = list(map(lambda x: (self.avaliar(x), estados.index(x)), estados))

        # evaluated_states.sort(reverse = True)
        evaluated_states.sort()

        n_melhores = evaluated_states[:n]

        # Retorna os n melhores na estrutura (sse, estado)
        return n_melhores

    # Retorna tupla (sse, estado) do melhor estado
    def melhorEstado(self, estados):
        melhor_estado = nMelhores(estados, 1)
        if melhor_estado:
            return melhor_estado[0]
        return []
    
    # Executa a busca pelo metodo de busca passado
    def busca(self, estado, metodoBusca, tempo, **argumentos):
        if argumentos:
            metodoBusca(self, estado, tempo, **argumentos)
        else:
            metodoBusca(self, estado, tempo)


    # PROBLEMA: SIMULATED ANNEALING
    # função que aceita ou não um vizinho de um estado
    def aceitarVizinho(self, estadoAtual, vizinho, t):
        if not self.valido(vizinho):
            return False
        elif self.melhorEstado([estadoAtual, vizinho]) == vizinho:
            return True
        else:
            valor_vizinho = self.avaliar(vizinho)
            valor_atual = self.avaliar(estadoAtual)
            # simmulated annealing calc
            p = 1/(m.exp((valor_atual - valor_vizinho)/t))
            p = p if p >= 0 else -p
            n = rand.random()
            return n < p

    # PROBLEMA: ALGORITMO GENETICO
    # função de seleção de sobreviventes
    def selecao(self, estados):
        pass

    # crossover de estados
    def crossover(self, estado1, estado2):
        pass

    # mutação num estado
    def mutacao(self, estado):
        pass

    # gera uma população a partir de um individuo
    def gerarPopulacao(self, populacao, tamanho):
        pass

    # retorna o melhor de uma geração e seu sse (OU SEM SSE?)
    def melhorDaGeracao(self, estados):
        melhor = self.melhorEstado(estados)
        if melhor:
            return melhor
        return self.estadoNulo()

    # PROBLEMA: GRASP
    def contrucaoGulosa(self, estado, m, seed, tempo):
        encerrou = False
        start = time.process_time

        melhor = self.estadoAleatorio



    #PROBLEMA: Branch and bound
    # função de estimativa
    def estimativa(self, estado, pessimista=True):
        pass











