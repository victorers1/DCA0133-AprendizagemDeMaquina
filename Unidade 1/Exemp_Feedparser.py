# -*- coding: utf-8 -*-

import feedparser

#Obtenção dos Feeds do G1
G1tec=feedparser.parse('http://pox.globo.com/rss/g1/tecnologia/')

print('Últimas notícias de tecnologia do G1:\n')
for post in G1tec.entries:
    print(post.title+'\n')

