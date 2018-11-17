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
        amemo = [np.array(entrada)]  # Matriz memória2. Guardaremos todas as saídas dos neurônios depois de passar para a função de ativação. Inclui a camada de entrada.
        
        #PASSADA DIRETA SALVANDO VALORES
        for v,p in zip(self.vieses, self.pesos):
            prox_entrada = []  # resultado desta camada é entrada para a próxima
            for i in range(len(v)):  # não pode ser len(p)
                z = np.dot(p[i,:], entrada)+v[i]  # Calcula soma ponderada com viés
                prox_entrada.append(z[0])  # 'z' é uma lista de um valor
            zmemo.append(np.array(prox_entrada))  # guarda valor antes de passá-la para função de ativ.
            entrada = ativacao(prox_entrada) # att. entrada
            amemo.append(np.array(entrada))  # guarda valores depois da ativação, não sei se será necessário ainda
        
        #converte tudo em np.array para usufruirmos de suas funções
        zmemo = np.array(zmemo)
        entrada = np.array(entrada)
        amemo = np.array(amemo)
        #print('ativ=\n{0}'.format(amemo))
        
        #PASSADA INVERSA, RETROPROPAGAÇÃO DO ERRO
        grad_p = [np.zeros(p.shape) for p in self.pesos]
        grad_v = [np.zeros(v.shape) for v in self.vieses]
        print('grad_v inicial= {}\n'.format(grad_v))
        #delta da camada de saída é calculado diferentemente dos outros
        delta = self.dcusto(gabarito, amemo[-1])*dativacao(zmemo[-1])  #conteúdo de 'amemo[-1]' é igual o de 'entrada'
        grad_v[-1] = delta.transpose()
        for i in range(len(grad_p[-1][0])):  # Pega primeiro elemento dos pesos da última camada
            grad_p[-1][:,i] = (delta*amemo[-1]) # ou np.transpose()
        
        for l in range(2, self.num_camadas):  #-l é a camada que está sendo analizada
            z = zmemo[-l]  # Valores da camada em análize antes da ativação
            a = amemo[-l]  # As ativações da camada em análise
            p = self.pesos[-l+1]  # Recupero pesos da camada seguinte
            #print('z= {}; a= {}; p= {}'.format(z,a,p))
            # OBS: gradiente da camada seguinte já está salvo em 'delta'
            prox_delta = []  # lista com os próximos valores de delta
            for i in range(len(p[0])):  # cada iteração calcula o delta de um neurônio da camada l
                #print('appending to prox_delta=\n{}'.format(np.dot(p[:,i], delta)*dativacao(z[i])))
                prox_delta.append(np.dot(p[:,i], delta)*dativacao(z[i]))
                prox_delta = np.array(prox_delta).transpose()
            
            delta = np.array(prox_delta.copy())
            grad_v[-l] = delta.transpose()
            #print('grad_v[-{}]= {}\n'.format(l, grad_v[-l]))
            for i in range(len(grad_p[-l][0])):
                grad_p[-l][:,i] = (delta*a)
                
        #print('grad_p final=\n{}'.format(grad_p))
        for v in grad_v:
            v = v.transpose()
        print('grad_v final=\n{}'.format(grad_v))
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
        grad_p_acum = np.array([np.zeros(p.shape) for p in self.pesos])  # acumulado dos gradientes de várias iterações do bakcpropagation
        grad_v_acum = np.array([np.zeros(v.shape) for v in self.vieses])
        #print('grad_v_acum inicial=\n{}'.format(grad_v_acum))
        # acumulado dos gradientes de várias iterações do backpropagation
        for x,y in zip(mini_lote[0], mini_lote[1]):
            grad_p, grad_v = self.backpropagation(x,y)
            print('grad_v_acum=\n{0}'.format(grad_v_acum))
            #print('grad_v=\n{0}'.format(grad_v))
            grad_p_acum = [gp+gpa for gp,gpa in zip(grad_p, grad_p_acum)]  # Realiza soma elemento a elemento entre duas lista de mesmo formado
            grad_v_acum = [gv+gva for gv,gva in zip(grad_v, grad_v_acum)]
            
        
        #print('antg pesos=\n{}'.format(self.pesos))
        #print('antg viese=\n{}'.format(self.vieses))
        
        self.pesos = [peso-lr*gradp for peso,gradp in zip(self.pesos, grad_p_acum)]
        #self.vieses = [vies-lr*gradv for vies,gradv in zip(self.vieses, grad_v_acum)]
        #print('novo pesos=\n{}'.format(self.pesos))
        #print('novo vieses=\n{}'.format(self.vieses))

#GERANDO DADOS
theta=np.linspace(0, 20, 2)
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

a = np.array([[1,2],[2,2],[3,2]])
a.transpose()
b = np.array([[3,4,5],[4,3,2]])
np.dot(a,b)

a=np.array([1,2])
b=np.array([1,2,3])
c=np.array([a,a,a])
d=np.array([b,b])
pesos = np.array([c,d])
pesos[-1][:,0]
pesos
'''