from gensim import corpora, models, similarities
import os, string
from collections import defaultdict
from sklearn.cluster import KMeans
from bhtsne import bh_tsne as tsne
import bhtsne
import numpy as np
import time

W2Vec = models.Word2Vec

print "Modules loaded"

def load_stopwords(stopfile):
    with open(stopfile,'r') as f: 
        return  map(lambda w: w.strip(),  f.readlines())

"""
-------------------------------------------------------------
CONFIGURABLE PARAMETERS
-------------------------------------------------------------
"""

data_dir    = os.path.join(os.getcwd(),'data')
text_dir = os.path.join(os.getcwd(), "text_data")
terminals = ":,;"
output_csv = os.path.join(os.getcwd(),"visuals","data.csv");
stoplist  = load_stopwords("stopwords.txt")
WORDVEC_SIZE = 200
GOOGLE_DATA  = "GoogleNews-vectors-negative300.bin.gz"
NUM_KMEANS_CLUSTERS=10
""" 
-------------------------------------------------------------
-------------------------------------------------------------
"""
def load_data_folder(data_dir=data_dir):
    """
    Returns a hashmap of article names mapped to their actual text
    """
    articles = {}
    for article in os.listdir(data_dir):
        articles[article]=load_doc(os.path.join(data_dir, article))
    return articles

def load_doc(fpath):
    text=""
    with open(fpath, 'r') as f:
        text = f.readlines()
        text = map(lambda t: unicode(t, errors='replace'), text)
        text = map(parse_text, text)
        text = map(lambda inp: inp.lower(), text)
        if len(text)>1:
            text = reduce(lambda x,y:x+y, text)
    return text

def gen_load(data_dir):
    """Generator to return every sentence in a directory- to train the 
    text models"""
    for article in os.listdir(data_dir):
		fpath = os.path.join(data_dir,article)
		with open(fpath, 'r') as f:
			text = f.readlines() + [article.rpartition(".txt")[0]]
			text = map(parse_text, text)
			text = map(lambda line: line.split(), text)
			text = reduce(lambda x,y : x+y, text)
		yield text

def bag_of_wordify(text):
    if type(text)!=list:
        text = [text]
    bag_of_words = set([])
    for line in text:
        for token in line.split():
            if token not in stoplist:
                bag_of_words.add(token)
    return bag_of_words

def parse_text(text):
    out_str = ""
    for index, letter in enumerate(text):
        if letter=="s" and text[index-1]=="'":
            out_str+=" "
        elif letter in string.punctuation:
            out_str += " "
        else:
            out_str+=letter
    return out_str.lower()

def output_write(tags, valarray, clusters=None):
    for index in tags:
        fopen = open(output_csv,"w")
    wstr = ""
    fopen.write(wstr)
    wstr = "Name,Category,Type,XAxis,YAxis,\n"
    fopen.write(wstr)
    template = "{0}, {3}, JUNK, {1}, {2},\n"
    for index, key in enumerate(tags):
        if type(clusters)!=type(None): 
            wstr = template.format(key.replace("_"," "), valarray[index][0], 
                                   valarray[index][1], 
                                   "class_"+str(clusters[index]))
	else: 
            wstr = template.format(key.replace("_"," "),valarray[index][0], 
                                   valarray[index][1], "JUNK")
        fopen.write(wstr)
    fopen.close()
    print "DONE"
    return

def word2vectorize(w2v , articles):
	"""
	w2v : Word2Vec model object
	"""
	vector = np.zeros((len(articles), w2v.layer1_size) )
	for index, key in enumerate(articles):
		count=0
		for word in articles[key]:
			if word in w2v:
				vector[index]+=w2v[word]
				count+=1
		#vector[index]/=float(count)
		#normalize length of vector
		veclength = reduce(lambda x,y: x+y, 
                                   map(lambda x: x**2, 
                                       vector[index]))
		if veclength>0:
                    vector[index]/=veclength**0.5
	return vector

def google_model():
    w2v_model = W2Vec.load_word2vec_format(GOOGLE_DATA)
    return w2v_model

def kmeans_clusters(keys, data_matrix, n_clusters=NUM_KMEANS_CLUSTERS):
	""" KMeans overlayed on the word2vec features """
	km = KMeans(n_clusters=n_clusters, n_init=25, max_iter=1000)
	km.fit(data_matrix)
	return km.predict(data_matrix)

def LDA_train(wdict, articles, bow_corpus, num_topics=100):
    lda_model = models.ldamodel.LdaModel(id2word=wdict, num_topics=num_topics)
    lda_model.VAR_MAXITER = 5000
    lda_model.VAR_THRESH  = 0.001
    lda_model.update(bow_corpus)
    return lda_model

"""
Debug globals
"""
g_lda = None
g_vec = None
g_coords = None
g_clust  = None

def LDA2Vec(lda_model, bow_corpus):
    sparse = []
    for bow in bow_corpus:
        sparse+=[lda_model[bow]]
    vector = np.zeros((len(sparse),
                       lda_model.num_topics), 
                      dtype = np.float64)
    for index, datapoint in enumerate(sparse):
        for feat_index, value in datapoint:
            vector[index][feat_index]= value
    return vector


def LDA_run():
    global g_lda, g_vec, g_coords, g_clust
    articles = load_data_folder(text_dir)
    #articles = {key:bag_of_wordify(articles[key]) for key in articles}
    lda_keys = articles.keys()
    corpus = [bag_of_wordify(articles[key]) for key in articles]
    wdict  = corpora.Dictionary(corpus)
    bow_corpus = [wdict.doc2bow(text) for text in corpus]
    NUM_TOPICS = 100
    print "Training LDA model"
    lda_model = LDA_train(wdict, articles, bow_corpus, NUM_TOPICS)        
    
    print "Converting to vector representation"
    wordvec    =    LDA2Vec(lda_model, bow_corpus)
    g_vec = wordvec
    print "running tsne"
    coords= [coord for coord in bhtsne.bh_tsne(wordvec)]
    print "running kmeans"
    clusters = kmeans_clusters(articles.keys(), wordvec)
    #code.interact()
    output_write(articles.keys(), coords, clusters)
    return 

def word2vec_run():
    raw_art = load_data_folder(text_dir)
    s = []    
    for sentence in gen_load(text_dir):
	    s+=[sentence]    
    W2V = models.Word2Vec
    """
    All this should be configurablee
    """
    w2v = W2V(s, workers=4, window=5, min_count=3, size=WORDVEC_SIZE)
    wordvec = word2vectorize(w2v, raw_art)
    coords= [coord for coord in bhtsne.bh_tsne(wordvec)]
    clusters = kmeans_clusters(raw_art.keys(), wordvec)
    output_write(raw_art.keys(), coords, clusters)
    return coords



def w2v_builder(raw_art, dims, text_dir):
    s = []
    for sentence in gen_load(text_dir):
	    s+=[sentence]
    WORDVEC_SIZE=dims
    """
    All this should be configurable
    """
    print "Training Word2Vec model"
    w2v = models.Word2Vec(s, workers=4, 
                          window=5, min_count=3, 
                          size=WORDVEC_SIZE)
    wordvec = word2vectorize(w2v, raw_art)
    return wordvec


def lda_builder(articles, dims, text_dir):
    global g_lda
    lda_keys = articles.keys()
    corpus = [bag_of_wordify(articles[key])
              for key in articles]
    wdict  = corpora.Dictionary(corpus)
    bow_corpus= [wdict.doc2bow(text) 
                 for text in corpus]
    NUM_TOPICS = dims
    print "Training LDA model"
    lda_model = LDA_train(wdict, articles, 
                          bow_corpus, NUM_TOPICS)     
    g_lda = lda_model
    return  LDA2Vec(lda_model, bow_corpus)


def compose(builders, sizes):
    #function to compose a combination of features
    articles = load_data_folder(text_dir)
    keys = articles.keys()
    vector = -1
    vector_builders = [lambda x:fn(articles,x,text_dir)
                       for fn in builders]
    for vbuilder, size in zip(vector_builders, sizes):
        if type(vector)!= type(None): 
            vector = vbuilder(size)
        else: 
            vector = np.append(vector, 
                               vbuilder(size), 
                               axis=1)
    coords= [coord for coord in tsne(vector)]
    clusters = kmeans_clusters(keys, vector)
    output_write(keys, coords, clusters)
    return vector
        
all_builders = [w2v_builder, lda_builder]
		       	 
if __name__=="__main__":
    compose([w2v_builder], [50])
