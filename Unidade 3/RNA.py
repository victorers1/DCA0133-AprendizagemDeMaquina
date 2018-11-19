#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:49:26 2018

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt

def objetivo(n):
    return np.sin(n+np.sin(n)**2)


def linear(x):
    return x

def dlinear(x):
    return linear


def degrau(x):
    if type(x)!=list and type(x)!=np.ndarray:
        if x>= 0:
            return 1
        else:
            return 0
    else:
        retorno = []
        for i in x:
            if i>= 0:
                retorno.append(1)
            else:
                retorno.append(0)
        return retorno

def ddegrau(x):
    if type(x)!=list and type(x)!=np.ndarray:
        if x!= 0:
            return 1
        else:
            return 0
    else:
        retorno = []
        for i in x:
            if i!= 0:
                retorno.append(1)
            else:
                retorno.append(0)
        return retorno


def sigmoid(x):
    '''Função de ativação'''
    if type(x) != list and type(x) != np.ndarray:
        retorno = 1.0/(1.0+np.exp(-x))
    else:
        retorno = []
        for i in x:
            retorno.append(1.0/(1.0+np.exp(-i)))
    return np.array(retorno)
    
def dsigmoid(x):
    '''Derivada da ativação'''
    if type(x) != list and type(x) != np.ndarray:
        retorno = sigmoid(x)*(1-sigmoid(x))
    else:
        retorno = []
        for i in x:
            retorno.append(sigmoid(i)*(1-sigmoid(i)))
    return np.array(retorno)


def relu(x):
    if type(x) != list and type(x) != np.ndarray:
        #print('retornando {}'.format(np.max([0,x])))
        return np.max([0,x]) 
    else:
        retorno = []
        for i in x:
            retorno.append(np.max([0,i]))
        #print('retornando {}'.format(retorno))
        return np.array(retorno)

def drelu(x):
    #print('tipo de {}= {}'.format(x, type(x)))
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
        return  np.array(retorno)


class RedeNeural(object):
    '''Classe que implementa uma rede neural artificial.'''
    
    def __init__(self, camadas, ativacao, dativacao):
        '''
        camadas: lista com números de neurônios em cada camada. Exemplo: [2,3,2] gera uma rede com 2 entradas, 3 neurônios na camada escondida e 2 saídas
        ativacao: nome da função de ativação usada em todas camadas. Exemplos: 'linear', 'relu', 'sigmoid', 'softmax' 
        custo: string com nome da função custo. Exemplos: 'mse', 'msle'
        tam_lote: inteiro com tamanho do lote em cada época.
        '''
        self.num_camadas = len(camadas)
        self.camadas = camadas
        self.ativacao = ativacao; self.dativacao = dativacao
        self.vieses = [np.random.rand(y,1) for y in camadas[1:]]
        self.pesos = [np.random.randn(y,x) for x,y in zip(camadas[:-1], camadas[1:])]
        
        
    def adiante(self, entrada):
        '''Recebe uma entrada e retorna a saída da rede neural.
        entrada: lista de valores'''
        for v,p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                prox_entrada.append(self.ativacao(np.dot(p[i,:], entrada)+v[i]))
            entrada = prox_entrada.copy()
        return np.array(entrada)
    
    
    def fcusto(self, a,b):
        '''Função custo que mede o quão distante a saída está da resposta desejada. Usada para avalizar o desempenho da rede'''
        return 1/2*(np.array(a)-np.array(b))**2
        
    def dcusto(self, a,b):
        '''Derivada da função custo.'''
        return np.array(a)-np.array(b)
    
    
    def backpropagation(self, entrada, gabarito):
        '''Backpropagation é um algoritmo que visa calcular os pesos de uma rede
        a partir dos erros na saída da mesma. Aqui, só são calculados os gradientes 
        (deltas) de cada neurônio. Os pesos são efetivamente atualizados noutra função.'''
        zmemo = [np.array(entrada)]  # Matriz memória1. Guardaremos todas as saídas dos neurônios antes de passar para a função de ativação
        amemo = [np.array(entrada)]  # Matriz memória2. Guardaremos todas as saídas dos neurônios depois de passar para a função de ativação. Inclui a camada de entrada.
        print('zmemo=\n{}'.format(zmemo))
        print('amemo=\n{}'.format(amemo))
        #PASSADA DIRETA SALVANDO VALORES
        for v,p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                z = np.dot(p[i,:], entrada)+v[i]  # Calcula soma ponderada com viés
                prox_entrada.append(z[0])  # 'z' é uma lista de um valor
            zmemo.append(np.array(prox_entrada))  # guarda valor antes de passá-la para função de ativ.
            entrada = self.ativacao(prox_entrada) # att. entrada
            amemo.append(np.array(entrada))  # guarda valores depois da ativação, não sei se será necessário ainda
        print('entrada=\n{}'.format(entrada))
        #converte tudo em np.array para usufruirmos de suas funções
        zmemo = np.array(zmemo)
        entrada = np.array(entrada)
        amemo = np.array(amemo)
        #print('zmemo=\n{}'.format(zmemo))
        #print('amemo=\n{}'.format(amemo))
        #print('saida=\n{}'.format(entrada))
        #PASSADA INVERSA, RETROPROPAGAÇÃO DO ERRO
        grad_p = [np.zeros(p.shape) for p in self.pesos]
        grad_v = [np.zeros(v.shape) for v in self.vieses]
        
        #delta da camada de saída é calculado diferentemente dos outros
        delta = self.dcusto(gabarito, amemo[-1])*self.dativacao(zmemo[-1])  #conteúdo de 'amemo[-1]' é igual o de 'entrada'
        #print('delta=\n{}'.format(delta))
        grad_v[-1] = delta[np.newaxis].T
        for i in range(len(grad_p[-1][0])):  # Pega primeiro elemento dos pesos da última camada
            grad_p[-1][:,i] = delta*amemo[-1]
        
        #print('grad_p=\n{}'.format(grad_p))
        for l in range(2, self.num_camadas):  #-l é a camada que está sendo analizada
            z = zmemo[-l]  # Valores da camada em análize antes da ativação
            a = amemo[-l]  # As ativações da camada em análise
            p = self.pesos[-l+1]  # Recupero pesos da camada seguinte
            # OBS: gradiente da camada seguinte já está salvo em 'delta'
            prox_delta = []  # lista com os próximos valores de delta
            for i in range(len(p[0])):  # cada iteração calcula o delta de um neurônio da camada l
                prox_delta.append(np.dot(p[:,i], delta)*self.dativacao(z[i]))  # Liga um valor real ao final de 'prox_delta'
            
            delta = prox_delta.copy()
            grad_v[-l] = np.array(delta)[np.newaxis].T
            for i in range(len(grad_p[-l][0])):
                grad_p[-l][:,i] = (delta*a)
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
        grad_p_acum = [np.zeros(p.shape) for p in self.pesos]  # acumulado dos gradientes de várias iterações do bakcpropagation
        grad_v_acum = [np.zeros(v.shape) for v in self.vieses]
        # acumulado dos gradientes de várias iterações do backpropagation
        for x,y in zip(mini_lote[0], mini_lote[1]):
            grad_p, grad_v = self.backpropagation(x,y)
            grad_p_acum = [gp+gpa for gp,gpa in zip(grad_p, grad_p_acum)]  # Realiza soma elemento a elemento entre duas lista de mesmo formado
            grad_v_acum = [gv+gva for gv,gva in zip(grad_v, grad_v_acum)]
        self.pesos = [peso-lr/n*gradp for peso,gradp in zip(self.pesos, grad_p_acum)]
        self.vieses = [vies-lr/n*gradv for vies,gradv in zip(self.vieses, grad_v_acum)]
        #print('grad_p final=\n{}'.format(grad_p))
        
    
    def treina(self, dados, epocas, tam_lote, lr):
        '''
        Implementa o algoritmo do Gradiente Estocástico Descendente
        Atualiza os mini lotes até que todos tenham passado pelo backpropagation
        dados_treino: lista de tuplas com formato (amostra, gabarito) cada
        epocas: inteiro com número de épocas
        tam_lote: inteiro com tamanho de cada mini_lote
        lr: taxa de aprendizagem
        '''
        dados_treino = dados.copy()
        #print('dados_treino=\n{}'.format(dados_treino))
        n = len(dados_treino)
        for e in range(epocas):
            print('\nÉpoca {} de {}'.format(e+1, epocas))
            np.random.shuffle(dados_treino)
            mini_lotes = [dados_treino[k:k+tam_lote] for k in range(0, n, tam_lote)]
            progresso = '>'
            for lote in mini_lotes:
                #print('lote em análise=\n{}'.format(lote))
                print('\r[{}'.format(progresso), end='')
                self.atualiza_lote(lote, 0.01)
                progresso = '=' + progresso
            print(']', end='')
            
'''
#GERANDO DADOS
theta=np.linspace(0, 20, 100)
x1 = theta/4*np.cos(theta)
y1 = theta/4*(np.sin(theta))
x2 = (theta/4+.8)*np.cos(theta)
y2 = (theta/4+.8)*(np.sin(theta))
x_treino,y_treino=[],[]
for v1,v2 in zip(x1,y1):
    x_treino.append(np.array([v1,v2]))
    y_treino.append([1,0])
for v1,v2 in zip(x2,y2):
    x_treino.append(np.array([v1,v2]))
    y_treino.append([0,1])
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
dados = []  # lista de tuplas com o par (amostra, gabarito)
for i in range(len(x_treino)):
    dados.append((x_treino[i], y_treino[i]))
'''

n=100  # Índice temporal do valor que será previsto 
qtd_pts = 50  # quantos valores há em cada amostra
qtd_amostra = 500  # quantas amostras serão usadas para treinar
x_treino,y_treino = [],[]

for i in range(qtd_amostra):
    ind = np.arange(n-i-1-qtd_pts, n-i-1) # índices temporais
    x_treino.append(objetivo(ind))
    y_treino.append(objetivo(n-i-1))
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
dados = []  # lista de tuplas com o par (amostra, gabarito)
for i in range(len(x_treino)):
    dados.append((x_treino[i], y_treino[i]))

rn = RedeNeural([qtd_pts,128,128,128,1], sigmoid, dsigmoid)
saida = rn.adiante(x_treino[0]); print('saida=\n{0}'.format(np.round(saida)))
#rn.ativacao(np.array([-1,0,2]))
#rn.dativacao(np.array([-2, -1, 0, 1, 2, 3]))
#rn.backpropagation(np.array([0,0]), np.array([1,0]))
#rn.atualiza_lote(dados, 0.01)

rn.treina(dados=dados, epocas=30, tam_lote=10, lr=0.01)

saida = rn.adiante(x_treino[0]); print('\nsaida pós treino=\n{0}'.format(np.round(saida)))


custo=0
Hcusto=[]
for d in dados:
    result = rn.adiante(d[0])
    custo = rn.fcusto(result, d[1].reshape(-1, 1))
    Hcusto.append(np.sum(custo))


plt.plot(Hcusto)
plt.title('Análise de Custo')
plt.xlabel('Amostra')
plt.ylabel('Custo')
plt.show()

cont= 0
for d in dados:
    igual=True
    for i,j in zip(d[1], np.round(rn.adiante(d[0]))):
        if i!=j:
            igual=False
            break
    if igual:
        cont+=1
    #print('gab: {} -> res: {}'.format(d[1], np.round(rn.adiante(d[0])).transpose()))

print('qtd certos= {}/{}'.format(cont, len(dados)))




'''DEPÓSITO DE CÓDIGO

for x in x_treino:
    print('x= {0}'.format(x))

foo = (1, 2, 3)
bar = (4, 5, 6)
uniao = (foo,bar)

a = np.array([1,2,3,4])
a.transpose()

a = np.array([[1],[2]])
b = np.array([[1],[2]])
a == b
for (f, b) in zip(uniao[0], uniao[1]):
    print("f: "+str(f) + "; b: " + str(b))

a = np.array([[1,2],[2,2],[3,2]])
c = np.array([[2,1],[1,2],[2,3]])
a+c

import time

for i in range(10): 
    print('\r'+str(i),end='')
    time.sleep(1)
    
print('opa', end='\r', flush=True)
print('maria', flush=True)

b = np.array([[3,4,5],[4,3,2]])
b/2
np.dot(a,b)

a = '===='
print('[{:10}>]'.format(a))

from sys import stdout
from time import sleep
for i in range(1,20):
    stdout.write("\r%d" % i)
    stdout.flush()
    sleep(1)
stdout.write("\n") # move the cursor to the next line


def foo(func, a, b):
    return func(a,b)

def soma(a,b):
    return a+b
def subt(a,b):
    return a-b
    
print(foo(subt, 2,2))



p = '>'
p = '='+p
p
a=np.array([1,2])
b=np.array([1,2,3])
c=np.array([a,a,a])
d=np.array([b,b])
pesos = np.array([c,d])
pesos[-1][:,0]
pesos
'''