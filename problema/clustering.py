'''
Modulo da classe do problema de clustering.
'''
import numpy as np
from scipy.spatial.distance import cdist, sqeuclidean

from .utils import evaluate_state, get_diff_obs_state


class Clustering:
    '''
        Lida com a instãncia de um problema de clusterização.
    '''
    def __init__(self, data):
        '''
        valores do problema
            data: ndarray size M x N
                Cada linha da array é uma observação.
                As colunas são os atributos de cada observação
            num_obs: int
                Número de observações no dataset
            num_feat: int
                numero de features (atributos) no dataset
        '''
        self.data = data
        self.num_obs = len(data)
        self.num_feat = len(data[0])

    def generate_initial_centroids(self, k):
        '''
            Seleciona aleatoriamente K linhas para serem os centroides
        '''
        points = np.random.choice(self.data.shape[0], size=k, replace=False)
        return self.data[points].copy()

    def update_centroids(self, labels, k):
        """
        Parameters:
            labels : int ndarray
                array of the labels of the observations.
            k : int
                The number of centroids (codes).
        Returns:
            centroids: k x n ndarray
            new centroids matrix
            has_members : ndarray
                A boolean array indicating which clusters have members.
        """
        centroids = np.zeros((k, self.num_feat), dtype=self.data.dtype)

        # sum of the numbers of obs in each cluster
        obs_count = np.zeros(k, np.int)

        for i in range(self.num_obs):
            label = labels[i]
            obs_count[label] += 1
            centroids[label] += self.data[i]

        for i in range(k):
            cluster_size = obs_count[i]

            if cluster_size > 0:
                # Calculate the centroid of each cluster
                centroids[i] = centroids[i] / cluster_size

        # Return a boolean array indicating which clusters have members
        return centroids, obs_count > 0

    def assign_clusters(self, centroids):
        '''
        Parametros:
            centroids: ndarray size k x N
                Cada linha é um centroide
        Retornos:
            labels: ndarray size M
                Uma array contendo o index do cluster atribuido a cada observacao
            min_dists: ndarray size M
                Array contendo a distancia da i-ésima observação até o centroide mais proximo
        '''
        dists = cdist(self.data, centroids, 'sqeuclidean')
        labels = dists.argmin(axis=1)
        min_dists = dists[np.arange(len(labels)), labels]
        return labels, min_dists



    def __init_neighbourhood(self):
        '''
            Inicia uma vizinhança nula com N observacoes.
            O index da observação é guardado para propositos de ordenação.

        Estrutura:
                    (index, label, sse)
        nbhood = [
                    [  (0,   0,   0.0)],  #obs0
                    [  (1,   0,   0.0)],  #obs1
                    [  (2,   0,   0.0)]   #obs2
                            ....       ]
        '''

        index_col = np.arange(self.num_obs, dtype=np.int) #guarda o index mesmo se ordernar pelo sse
        label_col = np.zeros(self.num_obs, dtype=np.int)  # novo label do i-ésimo ponto
        sse_col = np.zeros(self.num_obs, dtype=np.float32)   # novo sse do i-ésimo estado

        neighbourhood = np.zeros(self.num_obs, dtype={'names':('index', 'label', 'sse'),
                                                'formats':('i4', 'i4', 'f4')})
        neighbourhood['index'] = index_col
        neighbourhood['label'] = label_col
        neighbourhood['sse'] = sse_col
        return neighbourhood

    def generate_nbhood(self, centroids, labels, min_dists):
        '''
            gera uma vizinhança do estado atual.
        '''
        sse = evaluate_state(min_dists)
        k = len(centroids)

        neighborhood = self.__init_neighbourhood()

        for i in range(self.num_obs):
            # Modifica o estado (label) do ponto i
            new_label = get_diff_obs_state(labels[i], k)
            neighborhood[i]['label'] = new_label

            old_distance = min_dists[i]
            new_distance = sqeuclidean(self.data[i], centroids[new_label])

            new_sse_for_state = sse - old_distance + new_distance
            neighborhood[i]['sse'] = new_sse_for_state

        return neighborhood


# # Descrição da instancia do problema
# def descricao(self):
#     pass TODO

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

# #PROBLEMA: Branch and bound
# # função de estimativa
# def estimativa(self, estado, pessimista=True):
#     pass
