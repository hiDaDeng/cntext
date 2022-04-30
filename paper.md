---
title: 'cntext: A simple Python tooks for text mining'
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

**cntext** is a text analysis package that provides semantic distance and semantic projection based on word embedding models. Besides,cntext also provides the traditional methods, such as word count , readability, document similarity, sentiment analysis, etc. The link of cntext repo is ``https://github.com/hidadeng/cntext`` .







# Statement of need

Human society is rich in cognitions, such as perception, thinking, attitude and emotion.As a carrier of ideas, texts not only reflect people's mental activities from the individual’s level but also reflects collective culture from the organizational and social level.In the field of social sciences, the main research path is mining personal mental activities and culture changes in society through text data[@tausczik2010psychological]



There are two common text analysis algorithms: dictionary-based method and word-embedding based method. The cntext library contains both types of algorithms.



## Dictionary-based method

We can count the occurrences of a certain type of word in the text based on an certain dictionary.For example,  using a emotional adjectives dictionary, such as NRC, we can count the occurrences of different emotional words in the text ,and then know the distribution of emotions in the text[@chen2021symbols] .



## Word-embeddings-based method

Compared with the dictionary-based method, word-embedding based method has more efficient word representation ability and retains rich semantic information.So the scope of research topics is more wider, including social prejudice (stereotype) [@garg2018word], cultural cognition [@kozlowski2019geometry; @kurdi2019relationship], semantic change[@hamilton2016diachronic; @rodman2020timely], individual judgment and decision-making psychology[@bhatia2019predicting] .  A large number of studies have been published in international journals, such as Nature, Science, PNAS, Academy Management Journal, American Sociological Review, Management Science, etc.



As far as we know, cntext is the only Python package that provides **semantic projection**. For instance, to recover the similarities in size among nouns in a certain category (e.g., animals), we project their representations onto the line that extends from the word-vector **small** to the word-vector **big**; and to order them according to how dangerous they are, we project them onto the line connecting **safe** and **dangerous** [@Grand2022SemanticPR].



# Features

Functional modules include

-  **stats.py** basic text information

  -  word count
  -  readability
  -  built-in dictionary
  -  sentiment analysis

-  **dictionary.py** build and extend dictionary(vocabulary)

  -  throught Sopmi(mutual information) algorithm
  -  expand dictionary throught Word2Vec algorithm

-  **similarity.py** document similarity

  -  cosine algorithm
  -  jaccard algorithm
  -  edit distance algorithm

-  **mind.py** digest cognitive(attitude、bias etc.) information

  -  tm.semantic_distance
  -  tm.semantic_projection

  

# Exmaples
## semantic distance

```python
>>>import cntext as ct
>>>#Note: this is a word2vec format model
>>>tm = ct.Text2Mind(w2v_model_path='glove_w2v.6B.100d.txt')
>>>engineer = ['program', 'software', 'computer']
>>>mans =  ["man", "he", "him"]
>>>womans = ["woman", "she", "her"]
>>>tm.sematic_distance(words=animals, 
>>>                    c_words1=mans, 
>>>                    c_words2=womans)
-0.38
```


0.38 means that in semantic space, engineer is closer to man, other than woman.


## semantic projection


```python
>>>animals = ['mouse', 'cat', 'horse',  'pig', 'whale']
>>>smalls = ["small", "little", "tiny"]
>>>bigs = ["large", "big", "huge"]
>>>
>>>tm.sematic_projection(words=animals, 
>>>                      c_words1=smalls, 
>>>                      c_words2=bigs)
[('mouse', -1.68),
 ('cat', -0.92),
 ('pig', -0.46),
 ('whale', -0.24),
 ('horse', 0.4)]
```


In size conception, mouse is smallest, horse is biggest.







# References
