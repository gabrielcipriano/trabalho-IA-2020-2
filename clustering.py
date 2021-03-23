import math as m
import random as rand

class TimeOutException(Exception):
    pass

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

        evaluated_states.sort(reverse = True)

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









