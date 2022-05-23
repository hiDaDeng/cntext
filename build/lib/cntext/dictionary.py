import jieba.posseg as pseg
import math,time
from gensim.models import word2vec
from pathlib import Path
from nltk.tokenize import word_tokenize
import multiprocessing
from collections import defaultdict
import numpy as np
import pandas as pd
from cntext.stats import load_pkl_dict
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from collections import Counter
from mittens import GloVe
from sklearn.feature_extraction.text import CountVectorizer
import jieba



STOPWORDS_zh = load_pkl_dict(file='STOPWORDS.pkl')['STOPWORDS']['chinese']
STOPWORDS_en = load_pkl_dict(file='STOPWORDS.pkl')['STOPWORDS']['english']


class SoPmi:
    """
    Only support chinese now.
    """
    def __init__(self, cwd, input_txt_file, seedword_txt_file, lang='chinese'):
        """
        :param cwd:  The current folder path of the code, just set cwd=os.getcwd() will be okay!
        :param input_txt_file:  The path of the corpus file
        :param seedword_txt_file:  the path of manually selected seed word file
        :param lang:  set the language for SoPmi
        """
        self.cwd = cwd
        self.text_file = input_txt_file
        self.seedword_txt_file = seedword_txt_file
        self.lang=lang


    '''分词'''
    def seg_corpus(self, train_data, seedword_txt_file):
        #Add words to the jieba dictionary to ensure that the segmentation can't cut the seed emotional words
        if self.lang=='chinese':
            sentiment_words = [line.strip().split('\t')[0] for line in open(seedword_txt_file, encoding='utf-8')]
            for word in sentiment_words:
                jieba.add_word(word)

            seg_data = list()
            count = 0
            for line in open(train_data, encoding='utf-8'):
                line = line.strip()
                count += 1
                if line:
                    seg_data.append([word.word for word in pseg.cut(line) if word.flag[0] not in ['u','w','x','p','q','m']])
                else:
                    continue
        elif self.lang=='english':
            pass
        return seg_data



    '''统计搭配次数'''
    def collect_cowords(self, seedword_txt_file, seg_data):
        def check_words(sent):
            if set(sentiment_words).intersection(set(sent)):
                return True
            else:
                return False

        cowords_list = list()
        window_size = 5
        count = 0
        sentiment_words = [line.strip().split('\t')[0] for line in open(seedword_txt_file, encoding='utf-8')]
        for sent in seg_data:
            count += 1
            if check_words(sent):
                for index, word in enumerate(sent):
                    if index < window_size:
                        left = sent[:index]
                    else:
                        left = sent[index - window_size: index]
                    if index + window_size > len(sent):
                        right = sent[index + 1:]
                    else:
                        right = sent[index: index + window_size + 1]
                    context = left + right + [word]
                    if check_words(context):
                        for index_pre in range(0, len(context)):
                            if check_words([context[index_pre]]):
                                for index_post in range(index_pre + 1, len(context)):
                                    cowords_list.append(context[index_pre] + '@' + context[index_post])
        return cowords_list


    def collect_candiwords(self, seg_data, cowords_list, seedword_txt_file):
        """
        Calculate the So-Pmi value
        :param seg_data:
        :param cowords_list:
        :param seedword_txt_file:
        :return:
        """
        def compute_mi(p1, p2, p12):
            '''Mutual information calculation function'''
            return math.log2(p12) - math.log2(p1) - math.log2(p2)
        '''统计词频'''
        def collect_worddict(seg_data):
            word_dict = dict()
            for line in seg_data:
                for word in line:
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1
            all = sum(word_dict.values())
            return word_dict, all
        '''统计词共现次数'''
        def collect_cowordsdict(cowords_list):
            co_dict = dict()
            candi_words = list()
            for co_words in cowords_list:
                candi_words.extend(co_words.split('@'))
                if co_words not in co_dict:
                    co_dict[co_words] = 1
                else:
                    co_dict[co_words] += 1
            return co_dict, candi_words
        '''收集种子情感词'''
        def collect_sentiwords(seedword_txt_file, word_dict):
            pos_words = set([line.strip().split('\t')[0] for line in open(seedword_txt_file, encoding='utf-8') if
                             line.strip().split('\t')[1] == 'pos']).intersection(set(word_dict.keys()))
            neg_words = set([line.strip().split('\t')[0] for line in open(seedword_txt_file, encoding='utf-8') if
                             line.strip().split('\t')[1] == 'neg']).intersection(set(word_dict.keys()))
            return pos_words, neg_words
        '''计算sopmi值'''
        def compute_sopmi(candi_words, pos_words, neg_words, word_dict, co_dict, all):
            pmi_dict = dict()
            for candi_word in set(candi_words):
                pos_sum = 0.0
                neg_sum = 0.0
                for pos_word in pos_words:
                    p1 = word_dict[pos_word] / all
                    p2 = word_dict[candi_word] / all
                    pair = pos_word + '@' + candi_word
                    if pair not in co_dict:
                        continue
                    p12 = co_dict[pair] / all
                    pos_sum += compute_mi(p1, p2, p12)

                for neg_word in neg_words:
                    p1 = word_dict[neg_word] / all
                    p2 = word_dict[candi_word] / all
                    pair = neg_word + '@' + candi_word
                    if pair not in co_dict:
                        continue
                    p12 = co_dict[pair] / all
                    neg_sum += compute_mi(p1, p2, p12)

                so_pmi = pos_sum - neg_sum
                pmi_dict[candi_word] = so_pmi
            return pmi_dict

        word_dict, all = collect_worddict(seg_data)
        co_dict, candi_words = collect_cowordsdict(cowords_list)
        pos_words, neg_words = collect_sentiwords(seedword_txt_file, word_dict)
        pmi_dict = compute_sopmi(candi_words, pos_words, neg_words, word_dict, co_dict, all)
        return pmi_dict


    def save_candiwords(self, pmi_dict):
        """
        save the so-pmi result
        :param pmi_dict:
        :return:
        """
        def get_tag(word):
            if word:
                return [item.flag for item in pseg.cut(word)][0]
            else:
                return 'x'
        pos_dict = dict()
        neg_dict = dict()
        Path(self.cwd).joinpath('output').mkdir(exist_ok=True)
        Path(self.cwd).joinpath('output', 'sopmi_candi_words').mkdir(exist_ok=True)
        negfile= Path(self.cwd).joinpath('output', 'sopmi_candi_words', 'neg.txt')
        posfile= Path(self.cwd).joinpath('output', 'sopmi_candi_words', 'pos.txt')

        f_neg = open(negfile, 'w+', encoding='utf-8')
        f_pos = open(posfile, 'w+', encoding='utf-8')

        for word, word_score in pmi_dict.items():
            if word_score > 0:
                pos_dict[word] = word_score
            else:
                neg_dict[word] = abs(word_score)

        for word, pmi in sorted(pos_dict.items(), key=lambda asd:asd[1], reverse=True):
            f_pos.write(word + ',' + str(pmi) + ',' + 'pos' + ',' + str(len(word)) + ',' + get_tag(word) + '\n')
        for word, pmi in sorted(neg_dict.items(), key=lambda asd:asd[1], reverse=True):
            f_neg.write(word + ',' + str(pmi) + ',' + 'neg' + ',' + str(len(word)) + ',' + get_tag(word) + '\n')
        f_neg.close()
        f_pos.close()
        return



    def sopmi(self):
        """
        main function(method) of SoPmi
        :return:
        """
        print('Step 1/4:...Preprocess   Corpus ...')
        start_time  = time.time()
        seg_data = self.seg_corpus(self.text_file, self.seedword_txt_file)
        print('Step 2/4:...Collect co-occurrency information ...')
        cowords_list = self.collect_cowords(self.seedword_txt_file, seg_data)
        print('Step 3/4:...Calculate   mutual information ...')
        pmi_dict = self.collect_candiwords(seg_data, cowords_list, self.seedword_txt_file)
        print('Step 4/4:...Save    candidate words ...')
        self.save_candiwords(pmi_dict)
        end_time = time.time()
        duration = end_time-start_time
        duration = round(duration, 2)
        print('Finish! used {0} s'.format(duration))






class W2VModels(object):
    def __init__(self, cwd, lang='chinese'):
        """
        initialize the W2VModels
        :param cwd:  current work directory
        :param lang:  set the language for W2VModels
        """
        self.cwd = cwd
        self.lang = lang
        self.start = time.time()


    def __preproces(self, documents):
        """
        对数据进行预处理,分词、去除停用词；   可以加单词同类型合并的
        :param documents:  文档列表
        :return:  清洗后的文档列表
        """
        docs = []
        if self.lang=='english':
            for document in documents:
                document = document.lower()
                document = [w for w in word_tokenize(document) if w not in STOPWORDS_en]
                docs.append(document)
            return docs
        elif self.lang=='chinese':
            for document in documents:
                words = jieba.cut(document)
                document = [w for w in words if w not in STOPWORDS_zh]
                docs.append(document)
            return docs
        else:
            assert 'Do not support {} language'.format(self.lang)




    def train(self, input_txt_file, vector_size=100, min_count=5, ngram=False):
        """
        train word2vec model for corpus
        :param input_txt_file:  corpus file path
        :param vector_size: dimensionality of the word vectors.
        :param min_count: Set the word to appear at least min_count times in the model
        :param ngram: whether to take the ngram case into account，default False
        :return:
        """

        documents = list(open(input_txt_file, encoding='utf-8').readlines())
        print('Step 1/4:...Preprocess   corpus ...')
        sents = self.__preproces(documents=documents)
        duration = int(time.time()-self.start)

        sentences = []
        if self.lang=='english' and ngram==True:
            phrase_model = Phrases(sents, min_count=10, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            for sent in sents:
                sentence = phrase_model[sent]
                sentences.append(sentence)
        elif self.lang=='chinese' and ngram==True:
            phrase_model = Phrases(sents, min_count=10, threshold=1)
            for sent in sents:
                sentence = phrase_model[sent]
                sentence = [w.replace('_', '') for w in sentence]
                sentences.append(sentence)
        else:
            sentences = sents

        print('Step 2/4:...Train  word2vec model\n            used   {} s'.format(duration))
        self.model = word2vec.Word2Vec(sentences, vector_size=vector_size, min_count=min_count, workers=multiprocessing.cpu_count())
        modeldir = Path(self.cwd).joinpath('output', 'w2v_candi_words')
        Path(self.cwd).joinpath('output').mkdir(exist_ok=True)
        Path(self.cwd).joinpath('output', 'w2v_candi_words').mkdir(exist_ok=True)
        modelpath = str(Path(modeldir).joinpath('w2v.model'))
        self.model.wv.save(modelpath)




    def __search(self, seedword_txt_file, n=50):
        seedwords = [w for w in open(seedword_txt_file, encoding='utf-8').read().split('\n') if w!='']
        self.similars_candidate_idxs = [] #the candidate words of seedwords
        dictionary = self.model.wv.key_to_index
        self.seedidxs = [] #transform word to index
        for seed in seedwords:
            if seed in dictionary:
                seedidx = dictionary[seed]
                self.seedidxs.append(seedidx)
        for seedidx in self.seedidxs:
            # sims_words such as [('by', 0.99984), ('or', 0.99982), ('an', 0.99981), ('up', 0.99980)]
            sims_words = self.model.wv.similar_by_word(seedidx, topn=n)
            #Convert words to index and store them
            self.similars_candidate_idxs.extend([dictionary[sim[0]] for sim in sims_words])
        self.similars_candidate_idxs = set(self.similars_candidate_idxs)




    def find(self, seedword_txt_file, topn=50):
        """
        According to the seed word file, select the top n words with the most similar semantics
        根据种子词txt文件, 选出语义最相近的topn个词
        :param seedword_txt_file:  seed word file path, only support .txt file now!
        :param topn: The number of candidate words to output, the default topn=50
        :return:
        """

        seedwordsname = seedword_txt_file.split('/')[-1].replace('.txt', '')
        seedwords = [w for w in open(seedword_txt_file, encoding='utf-8').read().split('\n') if w]
        simidx_scores = []

        print('Step 3/4:...Prepare similar candidates for each seed word in the word2vec model...')
        self.__search(seedword_txt_file)
        for idx in self.similars_candidate_idxs:
            score = self.model.wv.n_similarity([idx], self.seedidxs)
            simidx_scores.append((idx, score))
        simidxs = [w[0] for w in sorted(simidx_scores, key=lambda k:k[1], reverse=True)]

        simwords = [str(self.model.wv.index_to_key[idx]) for idx in simidxs][:topn]

        resultwords = []
        resultwords.extend(seedwords)
        resultwords.extend(simwords)

        txtdir = Path(self.cwd).joinpath('output', 'w2v_candi_words')
        Path(self.cwd).joinpath('output', 'w2v_candi_words').mkdir(exist_ok=True)
        candidatetxtfile = Path(txtdir).joinpath('{}.txt'.format(seedwordsname))
        with open(candidatetxtfile, 'w', encoding='utf-8') as f:
            for word in resultwords:
                f.write(word+'\n')
        duration = int(time.time()-self.start)
        duration = round(duration, 2)
        print('Step 4/4 Finish! Used {duration} s'.format(duration=duration))




def co_occurrence_matrix(documents, window_size=2, lang='chinese'):
    """
    Build a co-word matrix
    :param documents:  a list of documents
    :param window_size:  Window size, the default value is set to 2
    :param lang:  Language type, the default setting is "chinese"
    :return:
    """
    d = defaultdict(int)
    vocab = set()
    if lang == 'english':
        for document in documents:
            document = document.lower().split()
            for i in range(len(document)):
                token = document[i]
                vocab.add(token)  # add to vocab
                next_token = document[i + 1: i + 1 + window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]))
                    d[key] += 1

    elif lang =='chinese':
        for document in documents:
            document = list(jieba.cut(document))
            # iterate over sentences
            for i in range(len(list(document))):
                token = document[i]
                vocab.add(token)  # add to vocab
                next_token = document[i + 1: i + 1 + window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]))
                    d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab)  # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df








class Glove(object):
    def __init__(self, cwd, lang='chinese'):
        """
        initialize the Glove model

        :param lang: set language for Glove model
        :return:
        """
        self.lang=lang
        self.cwd = cwd



    def create_vocab(self, file, min_count=5):
        """
        create vocabulary

        :param file:  corpus file path, only support .txt file now!
        :params min_count: When building the vocabulary ignore terms that wordcount strictly higher than the given threshold
        :return:
        """
        self.glove_output_name = file.split('/')[-1].replace('.txt', '')
        print('Step 1/4: ...Create vocabulary for Glove.')
        text = open(file, encoding='utf-8').read()
        if self.lang=='chinese':
            words = list(jieba.cut(text))
            self.words = [w for w in words if w not in STOPWORDS_zh]
        else:
            words = text.split(' ')
            self.words = [w.lower() for w in words if w not in STOPWORDS_en]
        # 压缩vocab
        self.vocab = [k for (k, v) in Counter(self.words).items() if v >= min_count]
        return self.vocab


    def cooccurrence_matrix(self):
        """
        Create cooccurrence matrix

        :return:
        """
        print('Step 2/4: ...Create cooccurrence matrix.')
        cv = CountVectorizer(ngram_range=(1, 1),
                             vocabulary=self.vocab)

        docs = [' '.join(self.words)]
        X = cv.fit_transform(docs)
        Xc = (X.T * X)
        Xc.setdiag(0)
        self.coo_matrix = Xc.toarray()
        return self.coo_matrix


    def train_embeddings(self, vector_size=50, max_iter=25):
        """
        Train glove embeddings

        :params vector_size:  Dimensionality of the word vectors. Default: 50
        :params max_iter: Number of training epochs. Default: 25
        :return:
        """
        start = time.time()
        print('Step 3/4: ...Train glove embeddings. \n             Note, this part takes a long time to run')
        glove_model = GloVe(n=vector_size, max_iter=max_iter)
        embeddings = glove_model.fit(self.coo_matrix)
        self.glove_embeddings = np.column_stack((np.array(self.vocab), embeddings))

        end = time.time()
        duration = round(end-start, 2)
        print('Step 3/4: ... Finish! Use {} s'.format(duration))
        return self.glove_embeddings


    def save(self):
        """
        Save glove embeddings as a txt file. Note we will use gensim to convert the glove embeddings in word2vec format
        :return:
        """

        print('Step 4/4: ... Save the glove embeddings to a txt file')
        from gensim.scripts.glove2word2vec import glove2word2vec
        txtdir = Path(self.cwd).joinpath('output', 'Glove')
        Path(self.cwd).joinpath('output', 'Glove').mkdir(exist_ok=True)
        glove_file = Path(txtdir).joinpath('{}_glove.txt'.format(self.glove_output_name))
        w2v_file = Path(txtdir).joinpath('{}_w2v.txt'.format(self.glove_output_name))
        with open(glove_file, 'a+', encoding='utf-8') as f:
            res_text = ''
            for glove_embedding in self.glove_embeddings:
                res_text += ' '.join(glove_embedding) + '\n'
            f.write(res_text)
        glove2word2vec(glove_file, w2v_file)

        import os
        os.remove(glove_file)





















