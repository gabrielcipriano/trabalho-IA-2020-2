import math as m
import random as rand
import time
import numpy as np
from scipy.spatial.distance import cdist, sqeuclidean



class Clustering:
    def evaluate_state(self, min_dists):
        return(np.sum(min_dists))

    def generate_initial_centroids(self, data, k):
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
    def update_cluster_means(self, data, labels, k):
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
    def assign_clusters(self, data, centroids):
        dists = cdist(data, centroids, 'sqeuclidean')
        labels = dists.argmin(axis=1)
        min_dist = dists[np.arange(len(labels)), labels]

        return labels, min_dist

    # # Descrição da instancia do problema
    # def descricao(self):
    #     pass #TODO

    # # Retorna o SSE do estado atual
    # def avaliar(self, estado):
    #     pass

    # # Diz se um estado é valido (necessário?) //TODO
    # def valido(self, estado):
    #     pass

    # # retorna estado considerado nulo (Existe um estado nulo?) //TODO
    # def estadoNulo(self):
    #     pass

    # # Retorna um estado aleatorio (k centroides aleatorios)
    # def estadoAleatorio(self):
    #     pass
    
    # retorna os n melhores estados (centroides) numa lista de estados
    # def nMelhores(self, estados, n):
    #     evaluated_states = list(map(lambda x: (self.avaliar(x), estados.index(x)), estados))

    #     # evaluated_states.sort(reverse = True)
    #     evaluated_states.sort()

    #     n_melhores = evaluated_states[:n]

    #     # Retorna os n melhores na estrutura (sse, estado)
    #     return n_melhores

    # # Retorna tupla (sse, estado) do melhor estado
    # def melhorEstado(self, estados):
    #     melhor_estado = nMelhores(estados, 1)
    #     if melhor_estado:
    #         return melhor_estado[0]
    #     return []
    
    # # Executa a busca pelo metodo de busca passado
    # def busca(self, estado, metodoBusca, tempo, **argumentos):
    #     if argumentos:
    #         metodoBusca(self, estado, tempo, **argumentos)
    #     else:
    #         metodoBusca(self, estado, tempo)


    # # PROBLEMA: SIMULATED ANNEALING
    # # função que aceita ou não um vizinho de um estado
    # def aceitarVizinho(self, estadoAtual, vizinho, t):
    #     if not self.valido(vizinho):
    #         return False
    #     elif self.melhorEstado([estadoAtual, vizinho]) == vizinho:
    #         return True
    #     else:
    #         valor_vizinho = self.avaliar(vizinho)
    #         valor_atual = self.avaliar(estadoAtual)
    #         # simmulated annealing calc
    #         p = 1/(m.exp((valor_atual - valor_vizinho)/t))
    #         p = p if p >= 0 else -p
    #         n = rand.random()
    #         return n < p

    # # PROBLEMA: ALGORITMO GENETICO
    # # função de seleção de sobreviventes
    # def selecao(self, estados):
    #     pass

    # # crossover de estados
    # def crossover(self, estado1, estado2):
    #     pass

    # # mutação num estado
    # def mutacao(self, estado):
    #     pass

    # # gera uma população a partir de um individuo
    # def gerarPopulacao(self, populacao, tamanho):
    #     pass

    # # retorna o melhor de uma geração e seu sse (OU SEM SSE?)
    # def melhorDaGeracao(self, estados):
    #     melhor = self.melhorEstado(estados)
    #     if melhor:
    #         return melhor
    #     return self.estadoNulo()

    # PROBLEMA: GRASP

    def __get_another_obs_state(self, current_label, k):
        new_label = rand.randint(0, k-1)
        while new_label == current_label:
            new_label = rand.randint(0, k-1)
        return new_label

    def generate_neighborhood(self, data, centroids, labels, min_dists):
        sse = self.evaluate_state(min_dists)
        k = len(centroids)
        num_obs = len(data)

        neighborhood = np.array([
            np.arange(num_obs), # guardar o index mesmo se ordernar
            np.zeros(num_obs),  # novo label do i-ésimo ponto
            np.zeros(num_obs)   # novo sse do i-ésimo estado
        ])

        for i in range(num_obs):
            # Modifica o estado (label) do ponto i
            new_label = self.__get_another_obs_state(labels[i], k)
            neighborhood[1][i] = new_label

            old_distance = min_dists[i]
            new_distance = sqeuclidean(data[i], centroids[new_label])

            new_sse_for_state = sse - old_distance + new_distance
            neighborhood[2][i] = new_sse_for_state

        return neighborhood


    def contrucao_gulosa(self, estado, m, tempo):
        start = time.process_time()

        melhor = self.generate_initial_centroids(data, k)



    # #PROBLEMA: Branch and bound
    # # função de estimativa
    # def estimativa(self, estado, pessimista=True):
    #     pass











