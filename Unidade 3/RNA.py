#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:49:26 2018

@author: victor
"""

import numpy as np

def ativacao(x):
    '''Função sigmoid'''
    if type(x) != list:
        retorno = 1.0/(1.0+np.exp(-x))
    else:
        retorno = []
        for i in x:
            retorno.append(1.0/(1.0+np.exp(-i)))
    return np.array(retorno)
    
def dativacao(x):
    '''Derivada da sigmoid'''
    if type(x) != list:
        retorno = ativacao(x)*(1-ativacao(x))
    else:
        retorno = []
        for i in x:
            retorno.append(ativacao(i)*(1-ativacao(i)))
    return np.array(retorno)


class RedeNeural(object):
    '''Classe que implementa uma rede neural artificial.'''
    
    def __init__(self, camadas):
        '''
        camadas: lista com números de neurônios em cada camada. Exemplo: [2,3,2] gera uma rede com 2 entradas, 3 neurônios na camada escondida e 2 saídas
        ativacao: nome da função de ativação usada em todas camadas. Exemplos: 'linear', 'relu', 'sigmoid', 'softmax' 
        custo: string com nome da função custo. Exemplos: 'mse', 'msle'
        tam_lote: inteiro com tamanho do lote em cada época.
        '''
        self.num_camadas = len(camadas)
        self.camadas = camadas
        self.vieses = [np.random.rand(y,1) for y in camadas[1:]]
        self.pesos = [np.random.randn(y,x) for x,y in zip(camadas[:-1], camadas[1:])]
        #print('pesos iniciais=\n'+str(self.pesos))
        #print('vieses iniciais=\n'+str(self.vieses))
        
        
    def adiante(self, entrada):
        '''Recebe uma entrada e retorna a saída da rede neural.
        entrada: lista de valores'''
        #print('entrada mesmo=\n{}'.format(entrada))
        #print('pesos mesmos=\n{}'.format(self.pesos))
        #print('viese mesmos=\n{}'.format(self.vieses))
        for v,p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                #print('p[{},:]= {}; entrada= {}'.format(i, p[i,:], entrada))
                prox_entrada.append(ativacao(np.dot(p[i,:], entrada)+v[i]))
            entrada = prox_entrada.copy()
            #print('entrada=\n'+str(entrada))
        return np.array(entrada)
    
    
    def SGD(self, treino, epocas, tam_lote):
        pass
    
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
        ativ = [np.array(entrada)]  # Matriz memória2. Guardaremos todas as saídas dos neurônios depois de passar para a função de ativação. Inclui a camada de entrada.
        
        #PASSADA DIRETA SALVANDO VALORES
        for v,p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                z = np.dot(p[i,:], entrada)+v[i]  # Calcula soma ponderada com viés
                prox_entrada.append(z[0])  # 'z' é uma lista de um valor
            zmemo.append(np.array(prox_entrada))  # guarda valor antes de passá-la para função de ativ.
            entrada = ativacao(prox_entrada) # att. entrada
            ativ.append(np.array(entrada))  # guarda valores depois da ativação, não sei se será necessário ainda
        
        #converte tudo em np.array para usufruirmos de suas funções
        zmemo = np.array(zmemo)
        entrada = np.array(entrada)
        ativ = np.array(ativ)
        #print('ativ=\n{0}'.format(ativ))
        
        #PASSADA INVERSA REPASSANDO ERRO
        grad_p = [np.zeros(p.shape) for p in zmemo]
        grad_v = [np.zeros(v.shape) for v in self.vieses]
        delta = grad_p.copy()
        
        delta[-1] = self.dcusto(gabarito, entrada)
        grad_p[-1] = delta[-1]*dativacao(zmemo[-1])*ativ[-1]  # delta de cada neurônio na saída
        grad_v[-1] = delta[-1]*ativ[-1] # delta da última camada de neurônios
        for l in range(2, self.num_camadas+1):  #-l é a camada que está sendo analizada
            z = zmemo[-l]  # Recupero os valores da camada em análize
            p = self.pesos[-l+1]  # Recupero pesos desta iteração
            for i in range(len(p[0])):  # cada iteração atualiza um neurônio da camada l
                delta[-l][i] = np.dot(p[:,i], grad_p[-l+1])*dativacao(z[i])

        for i in range(2, len(self.vieses)+1): # salva gradiente dos vieses fora da camada de saída
            grad_v[-i] = delta[-i]*ativ[-i]

        for i in range(len(delta)-1):  # preenche gradiente dos pesos da camada 1 até a penúltima
            for j in range(len(delta[i])):
                grad_p[i][j] = delta[i][j]*ativ[i][j]
        
        return grad_p, grad_v
    
    
    def atualiza_lote(self, mini_lote, lr):
        '''Atualiza pesos e vieses da rede com base no gradiente calculado pela
        função backpropagation(). Não há valor de retorno, tudo é modificado 
        diretamente nas variáveis internas da classe.
        mini_lote: lista de tuplas contendo o par (amostra, gabarito). Ambas 
        coordenadas devem ser estruturas compatíveis com a rede pois não haverá
        verificações nem tratamento de erros.
        lr: taxa de aprendizagem. geralmente um valor no intervalo (0,1)
        '''
        grad_p_acum = [np.zeros(p.shape) for p in self.pesos]  # acumulado dos gradientes de várias iterações do bakcpropagation
        grad_v_acum = [np.zeros(v.shape) for v in self.vieses]
        # acumulado dos gradientes de várias iterações do backpropagation
        for x,y in zip(mini_lote[0], mini_lote[1]):
            grad_p, grad_v = self.backpropagation(x,y)
            #print('grad_p=\n{0}'.format(grad_p))
            grad_p_acum = [ngp+gp for ngp,gp in zip(grad_p, grad_p_acum)]  # Realiza soma elemento a elemento entre duas lista de mesmo formado
            grad_v_acum = [ngv+gv for ngv,gv in zip(grad_v, grad_v_acum)]
            print('grad_v_acum=\n{}'.format(grad_v_acum))
        self.pesos = [peso-lr*grad for peso, grad in zip(self.pesos, grad_p_acum)]
        self.vieses = [vies-lr*grad for vies, grad in zip(self.vieses, grad_v_acum)]
        #print('novo pesos=\n{}'.format(self.pesos))
        #print('novo viese=\n{}'.format(self.vieses))

#GERANDO DADOS
theta=np.linspace(0, 20, 5)
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

rn = RedeNeural([2,3,2])
saida = rn.adiante(x_treino[1]); print('saida=\n{0}'.format(saida))

#g_pesos, g_vieses = rn.backpropagation(x_treino[0], y_treino[0])

rn.atualiza_lote((x_treino,y_treino), 0.1)

saida = rn.adiante(x_treino[1]); print('saida=\n{0}'.format(saida))


'''DEPÓSITO DE CÓDIGO

for x in x_treino:
    print('x= {0}'.format(x))

foo = (1, 2, 3)
bar = (4, 5, 6)
uniao = (foo,bar)

for (f, b) in zip(uniao[0], uniao[1]):
    print("f: "+str(f) + "; b: " + str(b))

'''