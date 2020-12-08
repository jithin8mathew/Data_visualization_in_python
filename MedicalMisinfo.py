#!/usr/bin/env python
# coding: utf-8

# Developed originally from the following code

# https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
# https://github.com/DerwenAI/pytextrank 
# https://github.com/susanli2016/NLP-with-Python/blob/master/AutoDetect_COVID_FakeNews.ipynb
# https://towardsdatascience.com/automatically-detect-covid-19-misinformation-f7ceca1dc1c7

import time
from nltk.corpus import stopwords    
stop_words = set(stopwords.words('english'))

from nltk.tag import pos_tag
from nltk import word_tokenize
from collections import Counter
import pandas as pd


import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
start_time = time.time()

df = pd.read_csv('corona_fake.csv')

df.loc[df['label'] == 'Fake', ['label']] = 'FAKE'
df.loc[df['label'] == 'fake', ['label']] = 'FAKE'
df = df.sample(frac=1).reset_index(drop=True)
df.title.fillna('missing', inplace=True)
df.source.fillna('missing', inplace=True)

from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm
    
    def get_keywords(self, number=10):
        keywordList = []
        rankList = []
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            keywordList.append(key)
            rankList.append(value)
#             print(key + ' - ' + str(value))
            if i > number:
                break
        return keywordList, rankList
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


def TextRank(text):
    try:
        tr4w = TextRank4Keyword()
        tr4w.analyze(text, candidate_pos = ['NOUN', 'ADJECTIVE'], window_size=4, lower=False)
        l, v = tr4w.get_keywords(10)
        return l
    except Exception:pass


def TextRankValues(text):
    try:
        tr4w = TextRank4Keyword()
        tr4w.analyze(text, candidate_pos = ['NOUN', 'ADJECTIVE'], window_size=4, lower=False)
        l, v = tr4w.get_keywords(10)
        return list(v)
    except Exception:pass
    
df['keywords'] = df.apply(lambda row: TextRank(row['text']), axis=1)

df['TextRankvalues'] = df.apply(lambda row: TextRankValues(row['text']), axis=1)

df.dropna()

def wordCount(lst,txt):
    val=0
    for x in lst:
        val+= txt.count(x)
    avg = val/len(lst)  
    return avg

avg_appearance=[]
for index, row in df.iterrows(): 
    try:
        df['avg_app'].loc[index]= wordCount(row['keywords'], row['text']) 
    except Exception:pass

df = df.fillna(value=np.nan)
df['wordLength'] = pd.DataFrame(df['keywords'].map(lambda x: len(str(x))))

def first_occurance(kwrdLst, text):
    if isinstance(text, str):
        text = text.lower().split(' ')
        txtLn= len(text)
    else:
        pass
    l1, l2 = [], []
    
    if isinstance(kwrdLst, list):
        for kwrd in kwrdLst:
            try:
                FO = text.index(str(kwrd).lower())
                FO_freq = (FO / txtLn)
                l1.append(FO)
                l2.append(FO_freq)
            except Exception:
                l1.append(0)
                l2.append(0)
    else:
        return [0],[0]
    return l1, l2

FOF = []
for index, row in df.iterrows():
    F = (first_occurance(row['keywords'],row['text']))
    FOF.append(F[1])
    
df['F_frequency']= FOF

def freq(kwrdLst, text):
    if isinstance(text, str):
        text = text.lower().split(' ')
        txtLn= len(text)
    else:
        pass
    F1 = []
    if isinstance(kwrdLst, list):
        for kwrd in kwrdLst:
            F1.append(text.count(kwrd))
    return(F1)

feqLst=[]
for index, row in df.iterrows():
    F = (freq(row['keywords'],row['text']))
    feqLst.append(F)

df['keywordFrequency'] = feqLst
    
count=0
finList = []
for index, row in df.iterrows():
    try:
        fin =  row['TextRankvalues'] + row['F_frequency']+ row['keywordFrequency']
    except Exception:
        fin = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99] + [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99] + [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99]
        count+=1
    finList.append(fin)
df['final'] = finList
print('Number of error encountered ', count)


def label(inp):
    if inp.lower() == "true":
        return 1
    else:
        return 0

df['wordLength'] = pd.DataFrame(df['keywords'].map(lambda x: len(str(x))))


ll=[]
for index, row in df.iterrows():
    ll.append(row['final'])

n = np.empty([len(ll), len(ll[0])])
n.shape

###################################
###################################
###################################
# np.array(ll)
c1=0
for lst in ll:
    c2=0
    for vv in lst:
        n[c1][c2]= vv
        #print(c1,c2)
        c2+=1
    c1+=1

labeL = []
for index, row in df.iterrows():
    labeL.append(label(str(row['label'])))

df['label2']=labeL

n = np.nan_to_num(n)
X, y = n, df['label2']
from sklearn.preprocessing import StandardScaler

scaled_features = StandardScaler().fit_transform(X)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109) 

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
SVCclf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
SVCclf.fit(X_train, y_train)
y_pred = SVCclf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0,verbose=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

C_range=list(range(1,26))
acc_score=[]
accGNB_score=[]
accNN_score = []
for c in C_range:
    #svc = LinearSVC(dual=False, C=c)
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    gnb = GaussianNB()
    clf = Perceptron(tol=1e-3, random_state=0)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    scoresGNB = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
    scoresNN = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
    accGNB_score.append(scoresGNB.mean())
    accNN_score.append(scoresNN.mean())

C_values=list(range(1,26))
fig = go.Figure(data=go.Scatter(x=C_values, y=acc_score, name='SVM LC'))
fig.add_scatter(x=C_values, y=accGNB_score, mode='lines', name='Gaussian')
fig.add_scatter(x=C_values, y=accNN_score, mode='lines', name='Neural Net')
fig.update_layout(xaxis_title='Value of iteration through CV data',
                   yaxis_title='Cross Validated Accuracy', template='plotly_white',xaxis = dict(dtick = 1))
fig.show()

print("--- %s seconds ---" % (time.time() - start_time))



