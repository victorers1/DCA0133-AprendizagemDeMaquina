# -*- coding: utf-8 -*-

import feedparser

palRemov = ['"', 'sem', '\'','.', 'só', 'à', 'os', 'com', 'como', 'Há', 'há', 'para','tem', 'têm','uma' , 'uns', 'umas', 'um', 'por', 'das', 'é', 'É', 'se', 'dos', 'Em', 'A', 'a', 'o', 'ao','e','O', 'do', 'da', 'de', 'que', 'muito', 'na', 'no', 'até', 'após', 'são', 'mas', 'mais', 'menos', 'tem', 'pouco', 'nas', 'nos', 'em', ':', ';', ',', '!', '?', '-']
#Obtensão dos Feeds do G1
G1tec=feedparser.parse('http://pox.globo.com/rss/g1/tecnologia/')
G1pol=feedparser.parse('http://pox.globo.com/rss/g1/politica')
G1eco=feedparser.parse('http://pox.globo.com/rss/g1/economia/')

#Tratamento dos títulos (retirar vogais, preposições) deixando apenas palavras chaves
def filtra(titulo): #filtra recebe uma frase (manchete da notícia) e trata-a conforme explicado duranto o código
    titulo = titulo.lower()
    lista = titulo.split(' ')
    for i in range(len(lista)):
        lista[i] = lista[i].replace('\'', '') # tira todas aspas
    
    #retorna um vetor de 'palavra' tal que 'palavra' são todos os elementos de 'lista' que: 1-Não estão em palRemov; 2-Não contêm digitos
    return [palavra for palavra in lista if ((palavra not in palRemov) and palavra.isalpha())]

def naive_bayes(dicio, noticia): # recebe um dicionário e uma lista com as palavras da notícia
    qtdPal=len(dicio.keys()) # Quantidade de palavras em dicio sem considerar frequências
    qtdPalFreq=0 # Quantidade de palavras em dicio (considerando frequências)
    for pal in dicio:
        qtdPalFreq = qtdPalFreq+dicio[pal]
    prob=1 # Probabilidade inicial
    for palavra in noticia:
        if palavra in dicio.keys():
            prob = prob*(dicio[palavra]/(qtdPal+qtdPalFreq))
        else:
            prob = prob*(1/(qtdPal+qtdPalFreq))
    
    return prob

dicTec={} # dicio com frequência de cada palavra de tec. (d['palavra'] = freq de ocorrência)
dicPol={}
dicEco={}
listTec=[] # lista de strings com palavras de manchetes de tecnologia (após a primeira filtragem de palavras)
listPol=[]
listEco=[]
for posts in G1tec.entries: # para cada post baixado
    listTec = filtra(posts.title) # filtragem inicial de palavras
    for palavra in listTec: # para cada palavra do post atual
        if palavra in dicTec.keys(): # se a palavra não está cadastrada no dicio
            dicTec[palavra] = dicTec[palavra]+1 # se está cadastrada, incrementa frequência
        else:
            dicTec[palavra] = 2 # já fazemos a normalização de Laplace adiciona 1 em todas as frequências de palavras

for posts in G1pol.entries:
    listPol = filtra(posts.title) 
    for palavra in listPol: 
        if palavra in dicPol.keys(): 
            dicPol[palavra] = dicPol[palavra]+1
        else:
            dicPol[palavra] = 2 

for posts in G1eco.entries: 
    listEco = filtra(posts.title) 
    for palavra in listEco: 
        if palavra in dicEco.keys(): 
            dicEco[palavra] = dicEco[palavra]+1
        else:
            dicEco[palavra] = 2 

#Rebece nova notícia para classificar
listNoticia = filtra(input('Digite uma manchete relacionada ou a Tecnologia ou Política ou Economia: '))

#Testa se é de Tec.
probTec = naive_bayes(dicTec, listNoticia)

#Testa se é de Pol.
probPol = naive_bayes(dicPol, listNoticia)
        
#Testa se é de Eco.
probEco = naive_bayes(dicEco, listNoticia)

print('proTec: ' + str(probTec))
print('proPol: ' + str(probPol))
print('proEco: ' + str(probEco))

dicio = {probTec:'Tecnologia', probPol:'Política', probEco:'Economia'}
print(dicio[max(dicio.keys())]) # imprime o assunto cuja probabilidade é máxima
