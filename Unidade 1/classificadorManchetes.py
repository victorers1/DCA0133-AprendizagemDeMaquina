# -*- coding: utf-8 -*-

# Inspiração: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
# CLASSIFICADOR DE MANCHETES PELO MÉTODO DE NAIVE-BAYES

import feedparser
# Palavras e caracteres a serem removidas das manchetes
palRemov = ['"', 'sem', '\'', '.', 'só', 'à', 'os', 'com', 'como', 'Há', 'há', 'para', 'tem', 'têm', 'uma', 'uns',
            'umas', 'um', 'por', 'das', 'é', 'É', 'se', 'dos', 'Em', 'A', 'a', 'o', 'ao', 'e', 'O', 'do', 'da', 'de',
            'que', 'muito', 'na', 'no', 'até', 'após', 'são', 'mas', 'mais', 'menos', 'tem', 'pouco', 'nas', 'nos',
            'em', ':', ';', ',', '!', '?', '-', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w,' 'z']

#Obtenção dos Feeds do G1
G1tec=feedparser.parse('http://pox.globo.com/rss/g1/tecnologia/')
G1pol=feedparser.parse('http://pox.globo.com/rss/g1/politica')
G1eco=feedparser.parse('http://pox.globo.com/rss/g1/economia/')


def filtra(titulo):  # Filtra recebe uma frase (manchete da notícia) e trata-a conforme explicado durante o código
    titulo = titulo.lower()  # Remove letras maiúsculas
    lista = titulo.split(' ')  # Split separa uma string em várias substrings e guarda numa lista
    for i in range(len(lista)):
        lista[i] = lista[i].replace('\'', '')  # Tira aspas, esse caractere tem um tratamento especial pois geralmente vem escrito junto às palavras
    # Retorna uma LISTA de 'palavra' tal que 'palavra' são todos os elementos de 'lista' que 1-Não estão em palRemov; 2-Não contêm digitos
    return [palavra for palavra in lista if ((palavra not in palRemov) and palavra.isalpha())]


def naive_bayes(dicio, noticia):  # Recebe um dicionário e uma lista com as palavras da notícia. Retorna a probabilidade da notícia estar relacionada com o dicionário
    qtdPal=len(dicio.keys())  # Quantidade de palavras em dicio sem considerar a frequência de cada uma
    qtdPalFreq=0  # Quantidade de palavras em dicio considerando frequências
    for pal in dicio:
        qtdPalFreq = qtdPalFreq+dicio[pal]
    prob=1  # Probabilidade inicial
    for palavra in noticia:
        if palavra in dicio.keys():
            prob = prob*(dicio[palavra]/(qtdPal+qtdPalFreq))  # Cálculo da prob. de uma palavra com freq. não nula
        else:
            prob = prob*(1/(qtdPal+qtdPalFreq))  # Aplicamos a normalização de Laplace para amostrar com prob. 0
    return prob


def contagem(feedNoticia):  # Recebe um objeto do tipo 'feed' e cria a lista de palavras e o dicionário de frequências
    dicio = {}  # Dicionário a ser preenchido
    for post in feedNoticia.entries:  # Para cada notícia recebida
        listaPalavra = filtra(post.title)  # Cria uma lista de palavras filtradas do título da notícia
        for palavra in listaPalavra:  # Para cada palavra no título da notícia
            if palavra in dicio.keys():  # Se a palvra já é conhecida...
                dicio[palavra] = dicio[palavra]+1  # Incremente a freq. de ocorrência
            else:  # Se é uma palavra nova...
                dicio[palavra] = 2  # Adicione ao dicionário já com a freq. igual à 2, pois faremos a normalização de Laplace
    return dicio  # Retorna o dicionário com todas as frequências de ocorrência incrementadas em uma unidade 1 da real


#dicTec={} # Dicionário com frequência de cada palavra de tecnologia (dicTec['palavra'] = freq. de ocorrência)
#dicPol={} # Agora com freq. de cada palavra de política
#dicEco={} # E economia

dicTec = contagem(G1tec)  #
dicPol = contagem(G1pol)  # Criação dos dicionários
dicEco = contagem(G1eco)  #

while True:
    # Rebece nova notícia para classificar, filtra certas palavras e guarda numa lista
    listNoticia = filtra(input('Digite uma manchete relacionada ou a Tecnologia ou Política ou Economia: '))
    
    #Calcula probabilidade de ser Tec.
    probTec = naive_bayes(dicTec, listNoticia)
    #Calcula probabilidade de ser Pol.
    probPol = naive_bayes(dicPol, listNoticia)
    #Calcula probabilidade de ser Eco.
    probEco = naive_bayes(dicEco, listNoticia)
    # A SOMA DAS PROBABILIDADES NÃO É IGUAL À 1
    
    print('proTec: ' + str(probTec))  #
    print('proPol: ' + str(probPol))  # Os valores geralmente são bem pequenos
    print('proEco: ' + str(probEco))  #
    
    dicio = {probTec:'Tecnologia', probPol:'Política', probEco:'Economia'}
    print(dicio[max(dicio.keys())])  # Imprime o assunto cuja probabilidade é máxima
