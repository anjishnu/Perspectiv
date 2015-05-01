echo "Installing Python3"
sudo apt-get install python3
echo "Installing pip"
sudo apt-get install python3-pip
echo "Installing cherrypy"
sudo pip3 install cherrypy
echo "Installing nltk"
sudo pip3 install nltk
echo "Installing Gensim"
sudo pip3 install gensim
echo "Downloading stopwords"
python3 -m nltk.downloader stopwords
