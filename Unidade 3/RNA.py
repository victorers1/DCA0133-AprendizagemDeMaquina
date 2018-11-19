#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:49:26 2018
@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    '''Função de ativação'''
    if type(x) != list and type(x) != np.ndarray:
        retorno = 1.0 / (1.0 + np.exp(-x))
    else:
        retorno = []
        for i in x:
            retorno.append(1.0 / (1.0 + np.exp(-i)))
    return np.array(retorno)


def dsigmoid(x):
    '''Derivada da ativação'''
    if type(x) != list and type(x) != np.ndarray:
        retorno = sigmoid(x) * (1 - sigmoid(x))
    else:
        retorno = []
        for i in x:
            retorno.append(sigmoid(i) * (1 - sigmoid(i)))
    return np.array(retorno)


def relu(x):
    if type(x) != list and type(x) != np.ndarray:
        return np.max([0, x])
    else:
        retorno = []
        for i in x:
            retorno.append(np.max([0, i]))
        return np.array(retorno)


def drelu(x):
    # print('tipo de {}= {}'.format(x, type(x)))
    if type(x) != list and type(x) != np.ndarray:
        if x >= 0:
            return 1
        else:
            return 0
    else:
        retorno = []
        for i in x:
            if i >= 0:
                retorno.append(1)
            else:
                retorno.append(0)
        return np.array(retorno)


class RedeNeural(object):
    '''Classe que implementa uma rede neural artificial.'''

    def __init__(self, camadas, ativacao, dativacao):
        '''
        camadas: lista com números de neurônios em cada camada. Exemplo: [2,3,2] gera uma rede com 2 entradas, 3
        neurônios na camada escondida e 2 saídas
        ativacao: nome da função de ativação usada em todas camadas. Exemplos: 'linear', 'relu', 'sigmoid', 'softmax'
        custo: string com nome da função custo. Exemplos: 'mse', 'msle'
        tam_lote: inteiro com tamanho do lote em cada época.
        '''
        print('Rede construída. Arquitetura: {}'.format(camadas))
        self.num_camadas = len(camadas)
        self.camadas = camadas
        self.ativacao = ativacao;
        self.dativacao = dativacao
        self.vieses = [np.random.rand(y, 1) for y in camadas[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(camadas[:-1], camadas[1:])]

    def adiante(self, entrada):
        '''Recebe uma entrada e retorna a saída da rede neural.
        
        entrada: lista de valores'''
        for v, p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                prox_entrada.append(self.ativacao(np.dot(p[i, :], entrada) + v[i]))
            entrada = prox_entrada.copy()
        return np.array(entrada)

    def fcusto(self, a, b):
        '''Função custo que mede o quão distante a saída está da resposta desejada. Usada para avalizar o desempenho da
         rede'''
        return 1 / 2 * (np.array(a) - np.array(b)) ** 2

    def dcusto(self, a, b):
        '''Derivada da função custo. Usada do backpropagation.'''
        return np.array(a) - np.array(b)

    def backpropagation(self, entrada, gabarito):
        '''Backpropagation é um algoritmo que visa calcular os pesos de uma rede
        a partir dos erros na saída da mesma. Aqui, só são calculados os gradientes
        (deltas) de cada neurônio. Os pesos são efetivamente atualizados noutra função.'''
        zmemo = [np.array(
            entrada)]  # Matriz memória1. Guardaremos todas as saídas dos neurônios antes de passar para a função de ativação
        amemo = [np.array(
            entrada)]  # Matriz memória2. Guardaremos todas as saídas dos neurônios depois de passar para a função de ativação. Inclui a camada de entrada.
        # PASSADA DIRETA SALVANDO VALORES
        for v, p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                z = np.dot(p[i, :], entrada) + v[i]  # Calcula soma ponderada com viés
                prox_entrada.append(z[0])  # 'z' é uma lista de um valor
            zmemo.append(np.array(prox_entrada))  # guarda valor antes de passá-la para função de ativ.
            entrada = self.ativacao(prox_entrada)  # att. entrada
            amemo.append(np.array(entrada))  # guarda valores depois da ativação, não sei se será necessário ainda
        # converte tudo em np.array para usufruirmos de suas funções
        zmemo = np.array(zmemo)
        entrada = np.array(entrada)
        amemo = np.array(amemo)
        # PASSADA INVERSA, RETROPROPAGAÇÃO DO ERRO
        grad_p = [np.zeros(p.shape) for p in self.pesos]
        grad_v = [np.zeros(v.shape) for v in self.vieses]

        # delta da camada de saída é calculado diferentemente dos outros
        delta = self.dcusto(gabarito, amemo[-1]) * self.dativacao(
            zmemo[-1])  # conteúdo de 'amemo[-1]' é igual o de 'entrada'
        grad_v[-1] = delta[np.newaxis].T
        for i in range(len(grad_p[-1][0])):  # Pega primeiro elemento dos pesos da última camada
            grad_p[-1][:, i] = delta * amemo[-1]

        for l in range(2, self.num_camadas):  # -l é a camada que está sendo analizada
            z = zmemo[-l]  # Valores da camada em análize antes da ativação
            a = amemo[-l]  # As ativações da camada em análise
            p = self.pesos[-l + 1]  # Recupero pesos da camada seguinte
            # OBS: gradiente da camada seguinte já está salvo em 'delta'
            prox_delta = []  # lista com os próximos valores de delta
            for i in range(len(p[0])):  # cada iteração calcula o delta de um neurônio da camada l
                prox_delta.append(
                    np.dot(p[:, i], delta) * self.dativacao(z[i]))  # Liga um valor real ao final de 'prox_delta'

            delta = prox_delta.copy()
            grad_v[-l] = np.array(delta)[np.newaxis].T
            for i in range(len(grad_p[-l][0])):
                grad_p[-l][:, i] = (delta * a)
        return np.array(grad_p), np.array(grad_v)

    def atualiza_lote(self, mini_lote, lr):
        '''Atualiza pesos e vieses da rede com base no gradiente calculado pela
        função backpropagation(). Não há valor de retorno, tudo é modificado
        diretamente nas variáveis internas da classe.
        
        mini_lote: tuplas contendo o par (amostra, gabarito). Ambas coordenadas
        devem ser estruturas compatíveis com a rede pois não haverá verificação
        nem tratamento de erros.
        lr: taxa de aprendizagem. geralmente um valor no intervalo (0,1)
        '''
        n = len(mini_lote)
        grad_p_acum = [np.zeros(p.shape) for p in
                       self.pesos]  # acumulado dos gradientes de várias iterações do bakcpropagation
        grad_v_acum = [np.zeros(v.shape) for v in self.vieses]
        # acumulado dos gradientes de várias iterações do backpropagation
        for x, y in zip(mini_lote[0], mini_lote[1]):
            grad_p, grad_v = self.backpropagation(x, y)
            grad_p_acum = [gp + gpa for gp, gpa in zip(grad_p,
                                                       grad_p_acum)]  # Realiza soma elemento a elemento entre duas lista de mesmo formado
            grad_v_acum = [gv + gva for gv, gva in zip(grad_v, grad_v_acum)]
        self.pesos = [peso - lr / n * gradp for peso, gradp in zip(self.pesos, grad_p_acum)]
        self.vieses = [vies - lr / n * gradv for vies, gradv in zip(self.vieses, grad_v_acum)]

    def treina(self, dados, epocas, tam_lote, lr):
        '''
        Implementa o algoritmo do Gradiente Estocástico Descendente
        Atualiza os mini lotes até que todos tenham passado pelo backpropagation
        
        dados: lista de tuplas com formato (amostra, gabarito) cada
        epocas: inteiro com número de épocas
        tam_lote: inteiro com tamanho de cada mini_lote
        lr: taxa de aprendizagem
        '''
        dados_treino = dados.copy()
        n = len(dados_treino)
        for e in range(epocas):
            print('\nÉpoca {} de {}'.format(e + 1, epocas))
            np.random.shuffle(dados_treino)
            mini_lotes = [dados_treino[k:k + tam_lote] for k in range(0, n, tam_lote)]
            progresso = '>'
            for lote in mini_lotes:
                print('\r[{}'.format(progresso), end='')
                self.atualiza_lote(lote, lr)
                progresso = '=' + progresso
            print(']', end='')


# GERANDO DADOS
theta = np.linspace(0, 20, 100)
x1 = theta / 4 * np.cos(theta)
y1 = theta / 4 * (np.sin(theta))
x2 = (theta / 4 + .8) * np.cos(theta)
y2 = (theta / 4 + .8) * (np.sin(theta))
x_treino, y_treino = [], []
for v1, v2 in zip(x1, y1):
    x_treino.append(np.array([v1, v2]))
    y_treino.append([1, 0])
for v1, v2 in zip(x2, y2):
    x_treino.append(np.array([v1, v2]))
    y_treino.append([0, 1])
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
dados = []  # lista de tuplas com o par (amostra, gabarito)
for i in range(len(x_treino)):
    dados.append((x_treino[i], y_treino[i]))

rn = RedeNeural([2, 60, 60, 60, 2], sigmoid, dsigmoid)

rn.treina(dados=dados, epocas=30, tam_lote=10, lr=0.05)

custo = 0
Hcusto = []
for d in dados:
    result = rn.adiante(d[0])
    custo = rn.fcusto(result, d[1].reshape(-1, 1))
    Hcusto.append(np.sum(custo))  # custo é um vetor, então somamos suas componenetes

plt.plot(Hcusto)
plt.title('Análise de Custo')
plt.xlabel('Amostra')
plt.ylabel('Custo')
plt.show()

cont = 0
for d in dados:
    igual = True
    for i, j in zip(d[1], np.round(rn.adiante(d[0]))):
        if i != j:
            igual = False
            break
    if igual:
        cont += 1

    print('Desejado= {}; Resposta da rede= {}'.format(d[1], np.round(rn.adiante(d[0])).T[0]))

print('\nqtd de respostas certas= {}/{}'.format(cont, len(dados)))
