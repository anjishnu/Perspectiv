Perspectiv
==========

A data visualization library focusing on simplicity.

----

DEPENDENCIES:
- Python 2.7 
- Natural Language Tool Kit: http://www.nltk.org/
- Scikit-Learn:  http://scikit-learn.org/stable/
- Barnes-Hut TSNE Implementation: https://github.com/ninjin/barnes-hut-sne

Pain Points:
------------
- Compiling the BH_TSNE implementation to work on your is a bit of a pain - it uses a popular fortran library
called CBLAS which will usually be installed on your machine after 
- Download the BH_TSNE and in the compile_mac or compile_linux script - figure out where the
CBLAS libraries are located in your machine.
- Sometimes you need to change the script as well, not just the CBLAS location.
- The linux command - "locate cblas" might help.
- Recently while setting up bh_tsne on a linux dev server - I had to change the compile_linux 
file as follows:
----------------
CBLAS=/usr/lib64/atlas
g++ sptree.cpp tsne.cpp -o bh_tsne -O3 -I./CBLAS/include -L/usr/lib64/atlas -L./ -lcblas

# \/ This is the original 
#g++ sptree.cpp tsne.cpp -o bh_tsne -O3 -I./CBLAS/include -L./ -lcblas
----------------
^Note this should be used an indicator for the kind of changes you might have to make.


----------------------
Things to Do: A.K.A. List of ugly hacks
---------------------
(0) TODO: Improve the README
    It doesn't seem to be good enough right now, would love some feedback on how to improve it.

(1) TODO: Understanding BH_TSNE and Perplexity 
    (With very low number of articles, bhtsne errors out citing 'Perplexity too high', right now I'm dealing 
    with it using a exponential backoff technique to recompute things with a lower perplexity if 
    the code errors out - the elegantway would be to figure out what perplexity to use computationally)

(2) TODO:
    Disover limitations of JSON Request Size (Just how much data can we safely send via an Async JQuery request)
    Right we haven't hit any barriers related to request size, but if we want to handle massive data sizes
    this could be an issue.

(3) TODO:
    Optimize JSON Request - Currently the request is sending over a lot of metadata to the server - this can
    be reduced by at least 95% by incorporating consistent indices and sending back concise metadata. Should
    allow our prototype to handle a lot more data.

(4) TODO: Automate parameter sizes

    Right now the code recomputes word2vec features of an arbitary vector size. Somehow we need to determine
    this vector size computationally - same for number of kmeans clusters - 
    right now I'm using ceil(natural_log(x)).

(5) TODO: Make the second pass configurable

    The subset runs after  should be made configurable somehow.

(6) TODO: Caching and Precomputation:

    The subset computations can be quite slow for massive data sizes. We might be able to speed things up by
    caching some precomputed vectors, either on memory or to disk.

(7) Improve GUI, add navigation:
    
    The current GUI is very basic, need to add some buttons to make the recomputation more intuitive instead of
    trying to invoke it on every third click.