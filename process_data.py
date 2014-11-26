from gensim import corpora, models, similarities
import os, string
from collections import defaultdict
from sklearn.cluster import KMeans
from bhtsne import bh_tsne as tsne
import bhtsne
import numpy as np
import time
import nltk

W2Vec = models.Word2Vec

print "Modules loaded"

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r,  %2.2f sec' % \
              (method.__name__, te-ts)
        return result
    return timed

def load_stopwords(stopfile=None):        
    with open(stopfile,'r') as f: 
        return  map(lambda w: w.strip(),  f.readlines())

"""
-------------------------------------------------------------
CONFIGURABLE PARAMETERS
-------------------------------------------------------------
"""

data_dir    = os.path.join(os.getcwd(),'data')
text_dir = os.path.join(os.getcwd(), "text_data")
OUTPUT_CSV = os.path.join(os.getcwd(),"visuals","data","data.csv")
TEMP_CSV = os.path.join(os.getcwd(),"visuals","data","temp_data.csv")
WORDVEC_SIZE = 200
GOOGLE_DATA  = "GoogleNews-vectors-negative300.bin.gz"
NUM_KMEANS_CLUSTERS=10
VERBOSE = True


#stoplist  = load_stopwords("stopwords.txt")
stoplist = nltk.corpus.stopwords.words()
""" 
-------------------------------------------------------------
-------------------------------------------------------------
"""

#Utility function so that it's easy to turn printing on and off
def printl(*args):
    if VERBOSE: print args

def load_data_folder(data_dir=data_dir):
    """
    Returns a hashmap of article names mapped to their actual text
    """
    articles = {}
    for article in os.listdir(data_dir):
        articles[article]=load_doc(os.path.join(data_dir, article))
    return articles

def load_specific_data(filenames, data_dir=data_dir):
    filenames = set(map(lambda name: name.replace(' ','_'), filenames))
    articles = {}
    for article in os.listdir(data_dir):
        if article in filenames:
            articles[article]= load_doc(os.path.join(data_dir,article))
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

def load_raw(fpath):
    text = ""
    with open(fpath, 'r') as f:
        text = f.readlines()
    return text

def gen_load(data_dir, sset):
    """Generator to return every sentence in a directory- to train the 
    text models"""
    article_list = os.listdir(data_dir) if not sset else sset
    for article in article_list:
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

def output_write(tags, valarray, clusters=None, output_dir=OUTPUT_CSV):
    fopen = open(output_dir, 'w')
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
		for word in articles[key].split():
			if word in w2v:
				vector[index]+=w2v[word]
				count+=1
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
        printl(len(keys))
        if n_clusters > len(keys):
            #Not enough data
            import math
            n_clusters = int(math.ceil(math.log(len(keys))))
	km = KMeans(n_clusters=n_clusters, n_init=25, max_iter=1000)
	km.fit(data_matrix)
	return km.predict(data_matrix)

@timeit
def LDA_train(wdict, articles, corpus, num_topics=100):        
    lda_model = models.ldamodel.LdaModel(corpus, id2word=wdict, workers=3,
                                         num_topics=num_topics)
    #Multicore is running slower than regular LDA? Confusing.
    #lda_model = models.LdaMulticore(corpus, id2word=wdict, workers=3,
    #                                     num_topics=num_topics)
    lda_model.VAR_MAXITER = 5000
    lda_model.VAR_THRESH  = 0.001
    lda_model.update(corpus)
    return lda_model

"""
Debug globals
"""
g_lda = None
g_vec = None
g_coords = None
g_clust  = None

def LDA2Vec(lda_model, corpus):
    sparse = []
    for doc in corpus:
        sparse+=[lda_model[doc]]
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
    tfidf = models.tfidfmodel.TfidfModel(bow_corpus, normalize=True)
    tfidf_corpus  = [tfidf[doc] for doc in bow_corpus]
    NUM_TOPICS = 100
    
    printl( "Training LDA model")
    
    lda_model = LDA_train(wdict, articles, tfidf_corpus, NUM_TOPICS)        
    print ("Converting to vector representation")
    wordvec    =    LDA2Vec(lda_model, tfidf_corpus)
    g_vec = wordvec
    
    printl ("running tsne")
    
    coords= [coord for coord in bhtsne.bh_tsne(wordvec)]
    print "running kmeans"
    clusters = kmeans_clusters(articles.keys(), wordvec)
    output_write(articles.keys(), coords, clusters)
    return lda_model

def word2vec_run():
    raw_art = load_data_folder(text_dir)
    s = []    
    for sentence in gen_load(text_dir):
	    s+=[sentence]    
    W2V = models.Word2Vec
    """ All this should be configurable """
    w2v = W2V(s, workers=4, window=5, min_count=3, size=WORDVEC_SIZE)
    wordvec = word2vectorize(w2v, raw_art)
    coords= [coord for coord in bhtsne.bh_tsne(wordvec)]
    clusters = kmeans_clusters(raw_art.keys(), wordvec)
    output_write(raw_art.keys(), coords, clusters)
    return coords


def w2v_builder(raw_art, dims, text_dir, sset=None):
    s = []
    for sentence in gen_load(text_dir, sset):
	    s+=[sentence]
    """
    All this should be configurable
    """
    print "Training Word2Vec model"
    w2v = models.Word2Vec(s, workers=4, 
                          window=5, min_count=3, 
                          size=dims)
    wordvec = word2vectorize(w2v, raw_art)
    return wordvec, w2v


def lda_builder(articles, dims, text_dir, tfidf_on=True, sset=None):
    lda_keys = articles.keys()
    corpus = [bag_of_wordify(articles[key])
              for key in articles]
    wdict  = corpora.Dictionary(corpus)
    bow_corpus= [wdict.doc2bow(text)
                 for text in corpus]
    inp_corpus = bow_corpus
    if tfidf_on:
        tfidf = models.tfidfmodel.TfidfModel(bow_corpus, normalize=True)
        tfidf_corpus  = [tfidf[doc] for doc in bow_corpus]        
        inp_corpus = tfidf_corpus

    print "Training LDA model"
    lda_model = LDA_train(wdict, articles, 
                          inp_corpus, dims)     
    return  LDA2Vec(lda_model, inp_corpus), lda_model


def compose(builders, sizes, articles=None, output_dir=None):
    """
    Function to compose a combination of features
    """
    sset = None
    if not articles:
        articles = load_data_folder(text_dir)
    else: sset = set(articles.keys())

    keys = articles.keys()
    vector = -1
    trained_models = []

    for builder, size in zip(builders, sizes):
        tmp, model= builder(articles, size, text_dir, sset)
        trained_models+=[model]
        if type(vector)!= type(None):
            vector = tmp
        else:
            vector = np.append(vector, tmp, axis=1)

    print vector.shape
    tsne_success = False
    perplexity = 32
    while(not tsne_success and perplexity>0):
        try:
            printl("trying perplexity", perplexity)
            coords= [coord for coord in tsne(vector, 
                                             verbose=False, 
                                             perplexity=perplexity)]
            tsne_success = True
        except:
            perplexity = perplexity/2

    clusters = kmeans_clusters(keys, vector)
    output_write(keys, coords, clusters, output_dir = output_dir)
    return vector, trained_models
        
def subset_run(fnames):
    if len(fnames)==0:
        printl ("Not enough documents")
        return -1
    articles = load_specific_data(fnames, text_dir)
    if len(articles)==0:
        printl ("FAILURE")
        return
    printl ("articles loaded", len(articles))
    #Saving the output in another file to avoid confusion
    compose(global_builders, global_sizes, articles = articles, output_dir=TEMP_CSV)
    printl ("launching newly computed results on firefox")
    os.system("firefox "+ os.path.join(os.getcwd(),"visuals","temp.html"))
    return True

all_builders = [w2v_builder, lda_builder]


global_builders = [lda_builder]
global_sizes = [50]

if __name__=="__main__":

    compose(global_builders, global_sizes)
