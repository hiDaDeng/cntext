---
title: 'cntext: A Python package for text mining'
tags:
  - text mining
  - social science
  - management science
  - sentiment analysis
  - semantic analysis
authors:
  - name: Xudong Deng 
    orcid: 0000-0001-7391-8865
    affiliation: 1
  - name: Nan Peng 
    affiliation: 2

affiliations:
 - name: Manament Department, Harbin Institute of Technology, China
   index: 1
 - name: Faculty of Western Languages and Literatures, Harbin Normal Uninversity, China
   index: 2

date: 30 Aprill 2022
bibliography: paper.bib
---

# Summary

**cntext** is a text analysis package that provides semantic distance and semantic projection based on word embedding models. Besides,cntext also provides the traditional methods, such as word count , readability, document similarity, sentiment analysis, etc. The repo of cntext in github is ``https://github.com/hidadeng/cntext`` .







# Statement of need

Human society is rich in cognitions, such as perception, thinking, attitude and emotion.As a carrier of ideas, texts not only reflect people's mental activities from the individual’s level but also reflects collective culture from the organizational and social level.In the field of social sciences, the main research path is mining personal mental activities and culture changes in society through text data[@tausczik2010psychological]



There are two common text analysis algorithms: dictionary-based method and word-embedding based method. The cntext library contains both types of algorithms.



## Dictionary-based method

We can count the occurrences of a certain type of word in the text based on an certain dictionary.For example,  using a emotional adjectives dictionary, such as NRC, we can count the occurrences of different emotional words in the text ,and then know the distribution of emotions in the text[@chen2021symbols] .



## Word-embeddings-based method

Compared with the dictionary-based method, word-embedding based method has more efficient word representation ability and retains rich semantic information.So the scope of research topics is more wider, including social prejudice (stereotype) [@garg2018word], cultural cognition [@kozlowski2019geometry].  A large number of studies have been published in international journals, such as Nature, Science, PNAS, Academy Management Journal, American Sociological Review, Management Science, etc.



As far as we know, cntext is the only Python package that provides **semantic projection**. For instance, to recover the similarities in size among nouns in a certain category (e.g., animals), we project their representations onto the line that extends from the word-vector **small** to the word-vector **big**; and to order them according to how dangerous they are, we project them onto the line connecting **safe** and **dangerous** [Grand2022SemanticPR].





# Exmaples

## built-in dictionary

```
\usepackage{listings}
\begin{lstlisting}[language=Python]
import cntext as ct
# get the list of built-in dictionary pkl
ct.dict_pkl_list()
\end{lstlisting}
```


Run

```
\usepackage{listings}
\begin{lstlisting}
['DUTIR.pkl',
 'HOWNET.pkl',
 'sentiws.pkl',
 'ChineseFinancialFormalUnformalSentiment.pkl',
 'ANEW.pkl',
 'LSD2015.pkl',
 'NRC.pkl',
 'geninqposneg.pkl',
 'HuLiu.pkl',
 'AFINN.pkl',
 'ADV_CONJ.pkl',
 'LoughranMcDonald.pkl']
\end{lstlisting}
```



We list 12 pkl dictionary here, some of English dictionary listed below are organized from [quanteda.sentiment](https://github.com/quanteda/quanteda.sentiment)

| Built-in file                               | Dictionary                                                   | Lanuage         | Function                                                      |
| ------------------------------------------- | ------------------------------------------------------------ | --------------- | ----------------------------------------------------------- |
| DUTIR.pkl                                   | DUTIR                                                        | Chinese         | Seven categories of emotions                                |
| HOWNET.pkl                                  | Hownet                                                       | Chinese         | Positive,Negative                                           |
| sentiws.pkl                                 | SentimentWortschatz (SentiWS)                                | English         | Positive,Negative;<br>Valence                               |
| ChineseFinancialFormalUnformalSentiment.pkl | Chinese finance dictionary, contains formal,unformal,positive,negative | Chinese         | formal-pos,<br>formal-neg;<br>unformal-pos,<br>unformal-neg |
| ANEW.pkl                                    | Affective Norms for English Words (ANEW)                     | English         | Valence                                                     |
| LSD2015.pkl                                 | Lexicoder Sentiment Dictionary (2015)                        | English         | Positive,Negative                                           |
| NRC.pkl                                     | NRC Word-Emotion Association Lexicon                         | English         | fine-grained sentiment words;                               |
| HuLiu.pkl                                   | Hu&Liu (2004)                                                | English         | Positive,Negative                                           |
| AFINN.pkl                                   | ANEW                                                         | English         | valence                                                     |
| LoughranMcDonald.pkl                        | Accounting Finance LM Dictionary                             | English         | Positive and Negative emotion words in the financial field  |
| STOPWORDS.pkl                               |                                                              | English&Chinese | stopwordlist                                                |

load the pkl dictionary file


```
\usepackage{listings}
\begin{lstlisting}[language=Python]
import cntext as ct

print(ct.load_pkl_dict('NRC.pkl'))
\end{lstlisting}
```


Run


```
\usepackage{listings}
\begin{lstlisting}
{'NRC': {'anger': ['abandoned', 'abandonment', 'abhor', 'abhorrent', ...],
         'anticipation': ['accompaniment','achievement','acquiring', ...],
         'disgust': ['abject', 'abortion', 'abundance', 'abuse', ...],
         'fear': ['anxiety', 'anxious', 'apache', 'appalling', ...],
         ......
         }
\end{lstlisting}
```




## Sentiment analysis

```
\usepackage{listings}
\begin{lstlisting}[language=Python]
import cntext as ct

text = 'What a happy day!'

ct.sentiment(text=text,
             diction=ct.load_pkl_dict('NRC.pkl')['NRC'],
             lang='english')
\end{lstlisting}
```



Run


```
\usepackage{listings}
\begin{lstlisting}
{'anger_num': 0,
 'anticipation_num': 1,
 'disgust_num': 0,
 'fear_num': 0,
 'joy_num': 1,
 'negative_num': 0,
 'positive_num': 1,
 'sadness_num': 0,
 'surprise_num': 0,
 'trust_num': 1,
 'stopword_num': 1,
 'word_num': 5,
 'sentence_num': 1}
\end{lstlisting}
```





## Train Word2Vec


```
\usepackage{listings}
\begin{lstlisting}[language=Python]
import cntext as ct
import os

#Init model parametter
model = ct.W2VModels(cwd=os.getcwd(), 
                     lang='english')  
model.train(input_txt_file='data/w2v_corpus.txt')
\end{lstlisting}
```






## semantic distance

```
\usepackage{listings}
\begin{lstlisting}[language=Python]
import cntext as ct

#Note: this is a word2vec format model
tm = ct.Text2Mind(w2v_model_path='glove_w2v.6B.100d.txt')

engineer = ['program', 'software', 'computer']
mans =  ["man", "he", "him"]
womans = ["woman", "she", "her"]

tm.sematic_distance(words=animals, 
                    c_words1=mans, 
                    c_words2=womans)
\end{lstlisting}
```

Run


```
\usepackage{listings}
\begin{lstlisting}
-0.38
\end{lstlisting}
```



0.38 means that in semantic space, engineer is closer to man, other than woman.



## semantic projection


```
\usepackage{listings}
\begin{lstlisting}[language=Python]
animals = ['mouse', 'cat', 'horse',  'pig', 'whale']
smalls = ["small", "little", "tiny"]
bigs = ["large", "big", "huge"]

tm.sematic_projection(words=animals, 
                      c_words1=smalls, 
                      c_words2=bigs)
\end{lstlisting}
```

Run

```
\usepackage{listings}
\begin{lstlisting}
[('mouse', -1.68),
 ('cat', -0.92),
 ('pig', -0.46),
 ('whale', -0.24),
 ('horse', 0.4)]
\end{lstlisting}
```


In size conception, mouse is smallest, horse is biggest.







# References
