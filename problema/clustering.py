'''
Modulo da classe do problema de clustering.
    tipo de estrutura de dados:
        Numpy 2-dimensional arrays
'''
import random as rand
import numpy as np
# from scipy.spatial.distance import cdist, sqeuclidean
from scipy.spatial.distance import cdist


from .utils import evaluate_dists_state, generate_labels_nbhood, get_diff_obs_state


class Clustering:
    '''Lida com a instãncia de um problema de clusterização.
    '''
    def __init__(self, data):
        '''valores do problema:
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
        points = np.random.choice(self.num_obs, size=k, replace=False)
        return self.data[points].copy()

    def update_centroids(self, labels, k):
        """ Parameters:
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

    def update_centroids_safe(self, centroids, labels, k):
        """ Atualiza o estado da lista de centroides com base nas labels
            Difere da função update_centroids por corrigir internamente
            problemas de cluster sem membros
        """
        new_centroids, has_members = self.update_centroids(labels, k)
        # Caso algum centroide novo não possua membros, atribui a posicao anterior
        if not has_members.all():
            # Setting to previous centroid position
            new_centroids[~has_members] = centroids[~has_members]
        return new_centroids

    def assign_clusters(self, centroids):
        ''' Parametros:
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

    # GENTIC ALGORITHN
    def evaluate(self, state):
        """Retorna o sse de um centroide (state)"""
        min_dists = self.assign_clusters(state)[1]
        return evaluate_dists_state(min_dists)

    def gerar_populacao(self, populacao, t_pop, k):
        """Preenche uma população a partir do primeiro individuo da população dada
        """
        state = populacao[0]
        labels = self.assign_clusters(state)[0]

        while len(populacao) < t_pop:
            new_labels = labels.copy()
            rand_obs = rand.randrange(0, self.num_obs)

            new_labels[rand_obs] = get_diff_obs_state(labels[rand_obs], k)
            new_state, has_members = self.update_centroids(new_labels,k)

            if has_members.all():
                populacao.append(new_state)

    def selecao(self, states):
        """ função de selecao por roleta (mantendo um unico sobrevivente na população)
                1º: calcula as probabilidades de cada um sobreviver
                2º: calcula a faixa de sobrevivência
                3º: Roda a roleta
        """
        total = sum(list(map(self.evaluate, states)))
        percents = list(map(lambda s: (s, self.evaluate(s)/total),states))

        prob_ranges = list()
        low_bound = 0
        for s in percents:
            prob_ranges.append((s[0], low_bound, low_bound + s[1]))
            low_bound += s[1]

        n = rand.random()
        # n = rand.uniform(0,1)
        for prob in prob_ranges:
            if n >= prob[1] and n <= prob[2]:
                states.clear()
                states.append(prob[0])

    def mutacao(self, state):
        labels = self.assign_clusters(state)[0]
        k = len(state)

        # define aleatoriamente quantas mutacoes acontecerao nas labels (até 10)
        for _ in range(rand.randint(1,10)):
            rand_obs = rand.randrange(0, self.num_obs)
            labels[rand_obs] = get_diff_obs_state(labels[rand_obs], k)

        new_state = self.update_centroids_safe(state, labels, k)

        return new_state

    def melhor_da_geracao(self, states):
        num_pop = len(states)

        melhor = states[0].copy()
        melhor_sse = self.evaluate(states[0])

        for i in range(1, num_pop):
            sse = self.evaluate(states[i])
            if sse < melhor_sse:
                melhor = states[i].copy()
                melhor_sse = sse

        return melhor, melhor_sse

    #  SIMULATED ANNEALING
    def best_state(self, states):
        """ Retorna o melhor estado em uma lista de estados (centroides).
        """
        best = states[0] 
        best_value = np.inf

        for state in states:
            state_value = self.evaluate(state)
            if state_value < best_value:
                best = state
                best_value = state_value

        return best

    def  accept_neighbour(self, state, nb_labels, k, t):
        ''' função que aceita ou não um vizinho de um estado (centroide)
        '''
        nb_state = self.update_centroids_safe(state, nb_labels, k)

        _, state_min_dists = self.assign_clusters(state)
        nb_labels, nb_min_dists = self.assign_clusters(nb_state)

        state_sse = evaluate_dists_state(state_min_dists)
        nb_sse = evaluate_dists_state(nb_min_dists)

        if nb_sse < state_sse:
            return True, nb_state

        p = 1/(np.exp( -1*(state_sse - nb_sse)/t))
        p = p if p >= 0 else -p
        n = rand.random()

        if n < p:
            return True, nb_state
        else:
            return False, []


    def __init_evaluated_neighbourhood(self):
        ''' Inicia uma vizinhança nula com N observacoes.
            O index da observação é guardado para propositos de ordenação.

        Estrutura:
                    (index, label, sse)
        nbhood = [
                    [  (0,   0,   0.0)],  #obs0
                    [  (1,   0,   0.0)],  #obs1
                    [  (2,   0,   0.0)]   #obs2
                            ....       ]
        '''
        index_col = np.arange(self.num_obs, dtype=np.int)   #guarda o index mesmo se ordernar pelo sse
        label_col = np.zeros(self.num_obs, dtype=np.int)    # novo label do i-ésimo ponto
        sse_col = np.zeros(self.num_obs, dtype=np.float32)  # novo sse do i-ésimo estado

        neighbourhood = np.zeros(self.num_obs, dtype={'names':('index', 'label', 'sse'),
                                                'formats':('i4', 'i4', 'f4')})
        neighbourhood['index'] = index_col
        neighbourhood['label'] = label_col
        neighbourhood['sse'] = sse_col
        return neighbourhood

    def generate_evaluated_nbhood(self, centroids, labels, min_dists):
        ''' gera uma vizinhança do estado atual com cada label acompanhada de seu valor (sse).
        '''
        sse = evaluate_dists_state(min_dists)
        k = len(centroids)

        nbhood = self.__init_evaluated_neighbourhood()
        nbhood['label'] = generate_labels_nbhood(labels, k)

        aux = 0.
        for i, new_label in enumerate(nbhood['label']):
            old_distance = min_dists[i]
            # new_distance = sqeuclidean(self.data[i], centroids[new_label])

            # TODO: Testar linha abaixo
            # Calculando a distantancia euclideana de maneira eficiente
            aux = self.data[i] - centroids[new_label]
            new_distance = np.dot(aux, aux)

            new_sse = sse - old_distance + new_distance
            nbhood[i]['sse'] = new_sse

        return nbhood
