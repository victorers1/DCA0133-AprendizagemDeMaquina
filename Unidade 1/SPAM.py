# -*- coding: utf-8 -*-

# Inspiração: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
# CLASSIFICADOR DE SPAM PELO MÉTODO DE NAIVE-BAYES

# Palavras e caracteres a serem removidas das manchetes
palRemov = ['"', 'sem', '\'', '.', 'só', 'à', 'os', 'com', 'como', 'Há', 'há', 'para', 'tem', 'têm', 'uma', 'uns',
            'umas', 'um', 'por', 'das', 'é', 'É', 'se', 'dos', 'Em', 'A', 'a', 'o', 'ao', 'e', 'O', 'do', 'da', 'de',
            'que', 'muito', 'na', 'no', 'até', 'após', 'são', 'mas', 'mais', 'menos', 'tem', 'pouco', 'nas', 'nos',
            'em', ':', ';', ',', '!', '?', '-']


def filtra(texto):  # Filtra recebe um texto (conteúdo bruto do e-mail) e trata-a conforme explicado durante o código
    texto = texto.lower()  # Remove letras maiúsculas

    lista = texto.split(' ')  # Split separa uma string em várias substrings e guarda numa lista
    for i in range(len(lista)):  # TODO: Talvez o FOR  seja desnecessário. Verei mais tarde
        lista[i] = lista[i].replace('\'', '')  # Tira todas aspas, esse caractere tem um tratamento especial pois geralmente vem escrito junto às palavras
    # Retorna uma LISTA de 'palavra' tal que 'palavra' são todos os elementos de 'lista' que 1-Não estão em palRemov; 2-Não contêm digitos
    return [palavra for palavra in lista if ((palavra not in palRemov) and palavra.isalpha())]


def naive_bayes(dicio, email):  # Recebe um dicionário e uma lista com as palavras do email. Retorna a probabilidade do email estar relacionado com o dicionário
    qtdPal = len(dicio.keys())  # Quantidade de palavras em 'dicio' sem considerar a frequência de cada uma
    qtdPalFreq = 0  # Quantidade de palavras em 'dicio' considerando frequências
    prob = 1  # Probabilidade inicial

    for pal in dicio:
        qtdPalFreq = qtdPalFreq + dicio[pal]

    for palavra in email:
        if palavra in dicio.keys():
            prob = prob * (dicio[palavra] / (qtdPal + qtdPalFreq))  # Cálculo da prob. de uma palavra com freq. não nula
        else:
            prob = prob * (1 / (qtdPal + qtdPalFreq))  # Aplicamos a normalização de Laplace para amostrar com prob. 0
    return prob


def contagem(listEmail):  # Recebe uma lista com as palavras do email e cria o dicionário de frequências
    dicio = {}  # Dicionário a ser preenchido

    for palavra in listEmail:  # Para cada palavra no título da notícia
        if palavra in dicio.keys():  # Se a palvra já é conhecida...
            dicio[palavra] = dicio[palavra] + 1  # Incremente a freq. de ocorrência
        else:  # Se é uma palavra nova...
            dicio[palavra] = 2  # Adicione ao dicionário já com a freq. igual à 2, pois faremos a normalização de Laplace

    return dicio  # Retorna o dicionário com todas as frequências de ocorrência incrementadas em uma unidade 1 da real


dicSPAM = contagem()   # Criação dos dicionários
dicNSPAM = contagem()  # Serão lidos arquivos de texto com o conteúdo de emails

while True:
    # Rebece um exemplo de email para classificar, filtra certas palavras e guarda numa lista
    email = filtra(input('Digite o conteúdo de um email: '))

    probSPAM = naive_bayes(dicSPAM, email)  # Calcula probabilidade de ser SPAM

    probNSPAM = naive_bayes(dicNSPAM, email)  # Calcula probabilidade de ser NÃO ser SPAM
    # A soma das probabilidades não precisa ser igual a 1

    print('prob. spam: ' + str(probSPAM))  #
    print('prob. não spam: ' + str(probNSPAM))  # Os valores geralmente são bem pequenos
