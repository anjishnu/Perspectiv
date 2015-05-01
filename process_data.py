from gensim import corpora, models, similarities
import os, string
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE as sk_tsne
from bhtsne import bh_tsne as tsne
import bhtsne
import numpy as np
import time
import nltk
from functools import reduce

W2Vec = models.Word2Vec

print("Modules loaded")

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r,  %2.2f sec' % \
              (method.__name__, te-ts))
        return result
    return timed

def load_stopwords(stopfile=None):        
    with open(stopfile,'r') as f: 
        return  [w.strip() for w in f.readlines()]
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

"""
#SIMPLECACHE (Global Variables)
"""
cached_model = None
cached_w2v   = None
cached_lda   = None
cached_tsne  = None
"""
#ENDCACHE
"""

#Utility function so that it's easy to turn printing on and off
def printl(*args):
    if VERBOSE: print(args)

def load_data_folder(data_dir=data_dir):
    """
    Returns a hashmap of article names mapped to their actual text
    """
    articles = {}
    for article in os.listdir(data_dir):
        articles[article]=load_doc(os.path.join(data_dir, article))
    return articles

def load_specific_data(filenames, data_dir=data_dir):
    filenames = set([name.replace(' ','_') for name in filenames])
    articles = {}
    for article in os.listdir(data_dir):
        if article in filenames:
            articles[article]= load_doc(os.path.join(data_dir,article))
    return articles

def load_doc(fpath):
    text=""
    with open(fpath, 'r') as f:
        text = f.readlines()
        text = [str(t, errors='replace') for t in text]
        text = list(map(parse_text, text))
        text = [inp.lower() for inp in text]
        #if len(text)>1:
            #text = reduce(lambda x,y:x+y, text)
        text = "".join(text)   
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
            text = [article.rpartition(".txt")[0]]+f.readlines()
            text = list(map(parse_text, text))
            text = [line.split() for line in text]
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

def output_write(tags, valarray, clusters=None, output_dir=OUTPUT_CSV, indices=None):

    print(output_dir)
    fopen = open(output_dir, 'w')
    wstr = ""
    fopen.write(wstr)
    wstr = "Name,Category,Index,XAxis,YAxis,\n"
    fopen.write(wstr)
    """
    TYPE -> We'll use type as our Index
    """
    template = "{0},{3},{4},{1},{2},\n"
    for index, key in enumerate(tags):
        if type(clusters)!=type(None): 
            #For subset computation case
            if (indices!=None):
                index = indices[index]

            wstr = template.format(key, valarray[index][0], 
                                   valarray[index][1], 
                                   clusters[index],
                                   str(index))
        else:
            wstr = template.format(key, valarray[index][0], valarray[index][1], "JUNK")
        fopen.write(wstr)
    fopen.close()
    print("DONE")
    return

def word2vectorize(w2v , articles):
	"""
	w2v : Word2Vec model object
	"""
	vector = np.zeros((len(articles), w2v.layer1_size) )
	for index, key in enumerate(articles):
		count=0
                #print articles[key]
                #print "-----------BREAK------------"
		for word in articles[key].split():
			if word in w2v:
				vector[index]+=w2v[word]
				count+=1
		#normalize length of vector
		veclength = reduce(lambda x,y: x+y, 
                                   [x**2 for x in vector[index]])
		if veclength>0:
                    vector[index]/=veclength**0.5
	return vector

def google_model():
    w2v_model = W2Vec.load_word2vec_format(GOOGLE_DATA)
    return w2v_model

def kmeans_clusters(keys, data_matrix, n_clusters=NUM_KMEANS_CLUSTERS):
    """ KMeans overlayed on the word2vec features """
    if n_clusters > len(keys):
            #Not enough data
            n_clusters = int(math.ceil(math.log(len(keys))))
    km = KMeans(n_clusters=n_clusters, n_init=25, max_iter=1000)
    km.fit(data_matrix)
    return km.predict(data_matrix)

@timeit
def LDA_train(wdict, articles, corpus, num_topics=100):      
    lda_model = models.ldamodel.LdaModel(corpus, id2word=wdict,
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
    lda_keys = list(articles.keys())
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
    print("running kmeans")
    if(classes==None):
        clusters = kmeans_clusters(list(articles.keys()), wordvec)
    else:
        #Avoid kmeans if classes are already provided
        clusters = classes
    output_write(list(articles.keys()), coords, clusters)
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
    clusters = kmeans_clusters(list(raw_art.keys()), wordvec)
    output_write(list(raw_art.keys()), coords, clusters)
    return coords


def w2v_builder(raw_art, dims, text_dir, sset=None):
    s = []
    for sentence in gen_load(text_dir, sset):
	    s+=[sentence]
    """
    All this should be configurable
    """
    print("Training Word2Vec model")
    w2v = models.Word2Vec(s, workers=4, 
                          window=5, min_count=3, 
                          size=dims)
    wordvec = word2vectorize(w2v, raw_art)
    return wordvec, w2v

def google_builder(articles, dims, text_dir):
    print("Using Google's model")
    dims = 300 #Constant because we are working with Google vectors
    w2v = google_model()
    wordvec = word2vectorize(w2v, articles)
    return wordvec, w2v

def lda_builder(articles, dims, text_dir, tfidf_on=True, sset=None):
    lda_keys = list(articles.keys())
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

    print("Training LDA model")
    lda_model = LDA_train(wdict, articles, 
                          inp_corpus, dims)     
    return  LDA2Vec(lda_model, inp_corpus), lda_model


def compose(builders, sizes, articles=None, output_dir=OUTPUT_CSV,
            vector=None, keys=None, indices=None, sset=None,
            classes=None):
    """
    Function to compose a combination of features
    """
    global gvec

    
    if (vector==None or keys==None):
        #If vector isn't provided
        sset = None
        if not articles:
            articles = load_data_folder(text_dir)
            gvec = articles
        else: sset = set(articles.keys())
        keys = list(articles.keys())
        vector = -1
        trained_models = []

        for builder, size in zip(builders, sizes):
            tmp, model= builder(articles, size, text_dir, sset)
            trained_models+=[model]
            if type(vector)!= type(None):
                vector = tmp
            else:
                vector = np.append(vector, tmp, axis=1)
        print(vector.shape)

    gvec = vector
    tsne_success = False
    perplexity = 32
    while(not tsne_success and perplexity>0):
        try:
            printl("trying perplexity", perplexity)
            coords= [coord for coord in tsne(vector, 
                                             verbose=True, 
                                             perplexity=perplexity)]
            tsne_success = True
        except Exception as e:
            print(type(e)) 
            print(e.message)
            perplexity = perplexity/2

    if (tsne_success == False):
        print("T-SNE failed - all perplexity settings not working") 
        print("Use Scikit-Learn's implementation of TSNE")
        perplexity = 32
        while(not tsne_success and perplexity>0):
            try:
                model = sk_tsne(n_components=2, random_state=0,
                                perplexity = perplexity)
                t1 = time.time()
                output = model.fit_transform(vector)
                coords = [pair for pair in output]
                t2 = time.time()
                print("Time taken:", t2-t1, "seconds")
                tsne_success = True
            except Exception as e:
                print(type(e))
                print(e)
                print(e.message)
                perplexity/=2
                
    if (classes == None):
        clusters = kmeans_clusters(keys, vector)
    else: 
        clusters = classes
    
    output_write(keys, coords, clusters=clusters, indices=indices,
                 output_dir = output_dir)
    
    """Caching"""
    if(not sset):
        global cached_model, cached_tsne
        cached_model = vector
        cached_tsne  = coords
    return vector
        
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
    compose(global_builders, global_sizes, 
            articles = articles, output_dir=TEMP_CSV)
    printl ("launching newly computed results on firefox")
    os.system("firefox "+ os.path.join(os.getcwd(),"visuals","temp.html"))
    return True

def subset_run_mem(indices, fnames, classes=None):
    if len(indices)==0:
        print("Not enough documents")
    if (cached_model!=None):
        n_dims = cached_model.shape[1]
        vector = np.zeros((len(indices), n_dims))
        count = 0
        for index in indices:
            vector[count]+=cached_model[index]
            count+=1
        # New vector has been computed from old one.
        compose(global_builders, global_sizes, 
                vector = vector, keys=fnames,
                output_dir=TEMP_CSV, sset=True,
                classes = classes)
        printl ("launching newly computed results on firefox")
        os.system("firefox "+ os.path.join(os.getcwd(),"visuals","temp.html"))
    else:
        subset_run(fnames)
            
            
        
all_builders = [w2v_builder, lda_builder]


global_builders = [w2v_builder]
global_sizes = [50]

if __name__=="__main__":
    compose(global_builders, global_sizes)
