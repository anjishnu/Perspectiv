import web
import json
import process_data as pd
from argparse import ArgumentParser
import os

urls = (
    '/', 'index',
    '/post', 'pind'
)

class index:
    def POST(self):
        i = web.input()
        print i
        print zip(i.keys(), i.values())
        return "Hello"

class pind:
    def POST(self):
        #i= web.input()
        text = web.data().encode('utf8')     
        json_obj = json.loads(text)
        print "JSON OBJECT:\n\n\n",json_obj
        filenames, indices, classes = clean_json(json_obj)
        for name in filenames:
            print name
        if pd.subset_run_mem(indices = indices,
                             fnames  = filenames,
                             classes = classes):
            return "halo"
        else:        return "fail"

    def OPTIONS(self):
        #i =  web.input()
        #print type(i), dir(i)
        return

def clean_json(dirty_json):

    dirty_json = dirty_json[0]
    name_of  = lambda x: x['__data__']['Name']
    index_of = lambda x: x['__data__']['Index']
    class_of = lambda x: x['__data__']['Category']

    return (map(name_of, dirty_json), 
            map(int, map(index_of, dirty_json)),
            map(class_of, dirty_json))

if __name__ == "__main__":
    parser = ArgumentParser()
    pd.compose(pd.global_builders, pd.global_sizes)
    #os.system("firefox visuals/index.html &")
    print "model shape", pd.cached_model.shape
    app = web.application(urls, globals())
    app.run()
