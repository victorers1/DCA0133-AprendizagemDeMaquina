# -*- coding: utf-8 -*-

# Inspiração: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
# Código desenvolvido para Python versão 2.7, versões futuras podem não funcionar
# CLASSIFICADOR DE SPAM PELO MÉTODO DE NAIVE-BAYES

from __future__ import division
import string


# Palavras e caracteres a serem removidas das manchetes
palRemov = ['sem', 'só', 'à', 'os', 'aos', 'com', 'como', 'Há', 'há', 'para', 'tem', 'têm', 'uma', 'uns', 'um', 'umas',
            'por', 'das', 'as', 'é', 'É', 'se', 'dos', 'desse', 'deste', 'dessa', 'desta', 'desses', 'destes', 'dessas',
            'destas', 'essas', 'essa', 'esse', 'esses', 'Em', 'ao', 'do', 'da', 'de', 'que', 'na', 'no', 'até', 'após',
            'são', 'mas', 'mais', 'menos', 'tem', 'pouco', 'nas', 'nos', 'em', 'pela', 'pelo', 'pelas', 'pelos', 'ou',
            'por']
for letra in string.ascii_letters:
    palRemov.append(letra)  # Adiciona todas as letras do alfabeto à 'palRemov'
pontRemov = '\:;,!?-".\'|/[]{}()*@#><~^+='


def filtra(texto):  # Filtra recebe um texto (conteúdo bruto do e-mail) e trata-a conforme explicado durante o código
    texto = texto.lower()  # Remove letras maiúsculas
    texto = texto.replace('\n', ' ')
    lista = texto.split(' ')  # Split separa uma string em várias substrings e guarda numa lista
    for i in range(len(lista)):  # Para cada palavra na lista...
        for ponto in pontRemov:  # Para cada caractere inválido...
            lista[i] = lista[i].replace(ponto, '') # Tira todos pontos, esses caracteres têm um tratamento especial pois geralmente vêm escritos junto às palavras
    # Retorna uma LISTA de 'palavra' tal que 'palavra' são todos os elementos de 'lista' que 1-Não estão em palRemov; 2-Não contêm digitos
    return [palavra for palavra in lista if((palavra not in palRemov) and palavra.isalpha())]


def naive_bayes(dicio, novo_email):  # Recebe um dicionário e uma lista com as palavras do email. Retorna a probabilidade do email estar relacionado com o dicionário
    qtdPal = len(dicio.keys())  # Quantidade de palavras em 'dicio' sem considerar a frequência de cada uma
    qtdPalFreq = 0  # Quantidade de palavras em 'dicio' considerando frequências
    prob = 1  # Probabilidade inicial

    for pal in dicio:
        qtdPalFreq = qtdPalFreq + dicio[pal]

    for palavra in novo_email:
        if palavra in dicio.keys():
            prob = prob * (dicio[palavra] / (qtdPal + qtdPalFreq))  # Cálculo da prob. de uma palavra com freq. não nula
        else:
            prob = prob * (1 / (qtdPal + qtdPalFreq))  # Aplicamos a normalização de Laplace para amostrar com prob. 0
    return prob


def contagem(end_arq):  # Recebe o endereço de um arquivo txt e cria um dicionário com suas palavras
    dicio = {}  # Dicionário a ser preenchido

    arq = open(end_arq, 'r')
    str = arq.read()
    list_email = filtra(str)

    for palavra in list_email:  # Para cada palavra no título da notícia
        if palavra in dicio.keys():  # Se a palvra já é conhecida...
            dicio[palavra] = dicio[palavra] + 1  # Incremente a freq. de ocorrência
        else:  # Se é uma palavra nova...
            dicio[palavra] = 2  # Adicione ao dicionário já com a freq. igual à 2, pois faremos a normalização de Laplace
    arq.close()
    return dicio  # Retorna o dicionário com todas as frequências de ocorrência incrementadas em uma unidade 1 da real


dicSPAM = contagem('C:/Users/victo/OneDrive/UFRN/Apredizagem de Maquina/Unidade 1/Lista/emails/spam.txt')  # Criação dos dicionários
dicNSPAM = contagem('C:/Users/victo/OneDrive/UFRN/Apredizagem de Maquina/Unidade 1/Lista/emails/naospam.txt')  # Serão lidos arquivos de texto com o conteúdo de emails


while True:
    # Rebece um exemplo de email para classificar, filtra certas palavras e guarda numa lista
    email = filtra(input('Digite o conteúdo de um email entre aspas: '))  # Entre aspas porque a versão usada é a 2.7

    probSPAM = naive_bayes(dicSPAM, email)  # Calcula probabilidade de ser SPAM
    probNSPAM = naive_bayes(dicNSPAM, email)  # Calcula probabilidade de ser NÃO ser SPAM
    # A soma das probabilidades não precisa ser igual a 1

    print('prob. spam: '     + str(probSPAM))  #
    print('prob. não spam: ' + str(probNSPAM))  # Os valores geralmente são bem pequenos

    dicConclusao = {probSPAM: 'SPAM', probNSPAM: 'NÃO SPAM'}
    print('Conclusão: '+dicConclusao[max(dicConclusao.keys())])
