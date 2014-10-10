from gensim import corpora, models, similarities
import os, string
from collections import defaultdict
from sklearn.cluster import KMeans
from bhtsne import bh_tsne as tsne
import numpy as np
import time

W2Vec = models.Word2Vec

print "Modules loaded"

def load_stopwords(stopfile):
    with open(stopfile,'r') as f: 
        return  map(lambda w: w.strip(),  f.readlines())

"""-------------------------------------------------------------
CONFIGURABLE PARAMETERS
-------------------------------------------------------------"""
data_dir    = os.path.join(os.getcwd(),'data')
text_dir = os.path.join(os.getcwd(), "text_data")
terminals = ":,;"
output_csv = os.path.join(os.getcwd(),"visual","data.csv");
stoplist  = load_stopwords("stopwords.txt")
WORDVEC_SIZE = 200
""" -------------------------------------------------------------
-------------------------------------------------------------"""


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
        text= f.readlines()
        text= map(parse_text, text)
        text= map(lambda inp: inp.lower(), text)
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
		for word in articles[key].split():
			if word in w2v:
				vector[index]+=w2v[word]
				count+=1
		#vector[index]/=float(count)
		#normalize length of vector
		veclength = reduce(lambda x,y: x+y, 
                                   map(lambda x: x**2, 
                                       vector[index]))

		vector[index]/=veclength**0.5
	return vector

def kmeans_clusters(keys, data_matrix):
	""" KMeans overlayed on the word2vec features"""
	km = KMeans(n_clusters=10, n_init=25, max_iter=1000)
	km.fit(data_matrix)
	return km.predict(data_matrix)

def mine_text():
    raw_art = load_data_folder(text_dir)
    s = []    
    for sentence in gen_load(text_dir):
	    s+=[sentence]    
    W2V = models.Word2Vec
    """
    All this should be configurable
    """
    w2v = W2V(s, workers=4, window=5, min_count=3, size=WORDVEC_SIZE)
    wordvec = word2vectorize(w2v, raw_art)
    coords= [coord for coord in bhtsne.bh_tsne(wordvec)]
    clusters = kmeans_clusters(raw_art.keys(), wordvec)
    output_write(raw_art.keys(), coords, clusters)
    return coords
				 
if __name__=="__main__":
    t1 = time.time()
    raw_art = load_data_folder(text_dir)
    s = []    
    for sentence in gen_load(text_dir):
	    s+=[sentence]    
    w2v_model = W2Vec(s, workers=4, window=5, min_count=3, size=WORDVEC_SIZE)
    wordvec = word2vectorize(w2v_model, raw_art)
    coords= [coord for coord in tsne(wordvec)]
    clusters = kmeans_clusters(raw_art.keys(), wordvec)
    output_write(raw_art.keys(), coords, clusters)
    print "Execution complete"
    print "Total execution time:", time.time()-t1, "seconds"
