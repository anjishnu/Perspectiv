Perspectiv
==========

A data visualization library focusing on simplicity.

----

DEPENDENCIES:
- Python 2.7 
- Natural Language Tool Kit: http://www.nltk.org/
- Scikit-Learn:  http://scikit-learn.org/stable/
- Gensim: http://radimrehurek.com/gensim/
- Barnes-Hut TSNE Implementation: https://github.com/ninjin/barnes-hut-sne
- Web.py: http://webpy.org/install

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
<br>CBLAS=/usr/lib64/atlas
<br>g++ sptree.cpp tsne.cpp -o bh_tsne -O3 -I./CBLAS/include -L/usr/lib64/atlas -L./ -lcblas
<br>
<br>#g++ sptree.cpp tsne.cpp -o bh_tsne -O3 -I./CBLAS/include -L./ -lcblas
<br>#Original code ^Note this should be used an indicator for the kind of changes you might have to make.

HOW TO RUN
-------------------
1) Put all your plain text data in Perspective/text_data/

2) Run:
   $ python server.py & 
   (This starts the server which will is necessary for recomputation

3) Run:
   $ python process_data.py 
   (This starts off the initial data processing and populates it in the visuals folder)

4) Run:
   $ firefox visuals/index.html
   This should open up the D3 visualization.

5) To recompute subsets. Use the mouse to click and drag over a selection area, click to select, and then 
   click one more time to spark off the recomputation.

----------------------
Things to Do: A.K.A. List of ugly hacks
---------------------
(0) TODO: Improve the README

    It doesn't seem to be good enough right now, would love some feedback on how to improve it.

(1) TODO: Understanding BH_TSNE and Perplexity 

    (With very low number of articles, bhtsne errors out citing 'Perplexity too high', right now I'm dealing 
    with it using a exponential backoff technique to recompute things with a lower perplexity if 
    the code errors out - the elegantway would be to figure out what perplexity to use computationally)

(2) TODO: What happens during JSON overflow?

    Disover limitations of JSON Request Size (Just how much data can we safely send via an Async JQuery request)
    Right we haven't hit any barriers related to request size, but if we want to handle massive data sizes
    this could be an issue.

(3) TODO: Optimize JSON Request using Indexing

    Currently the request is sending over a lot of metadata to the server - this can
    be reduced by at least 95% by incorporating consistent indices and sending back concise metadata. Should
    allow our prototype to handle a lot more data.

(4) TODO: Automate parameter sizes

    Right now the code recomputes word2vec features of an arbitary vector size. Somehow we need to determine
    this vector size computationally - same for number of kmeans clusters - 
    right now I'm using ceil(natural_log(x)).

(5) TODO: Make the second pass configurable

    The subset runs after  should be made configurable somehow.

(6) TODO: Caching and Precomputation: #DONE

    The subset computations can be quite slow for massive data sizes. We might be able to speed things up by
    caching some precomputed vectors, either on memory or to disk.

(7) Improve GUI, add navigation: #DONE

    The current GUI is very basic, need to add some buttons to make the recomputation more intuitive instead of
    trying to invoke it on every third click.
    
(8) LDA_Multicore Mystery:

      I'm not quite sure why, but Multicore LDA is running slower than vanilla LDA. Can't quite figure out why.
      Right now both are in the code and the single threaded version has been enabled.

(8) Code Clean and Object Orientation:

      Right now, a lot of the codebase is in a mess using global variables caches etc. I want to clean it up and convert it in
      into suitable objects to make the system more extensible. 
