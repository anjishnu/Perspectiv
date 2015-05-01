import json
import process_data as pd
from argparse import ArgumentParser
import os
import cherrypy


class Index(object): #
        @cherrypy.expose
        def index(self):
                return

class Post(object):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    def POST(self):
        json_obj = cherrypy.request.json
        print ("JSON OBJECT:\n", json_obj)
        filenames, indices, classes = clean_json(json_obj)
        for name in filenames:
            print (name)
        if pd.subset_run_mem(indices = indices,
                             fnames  = filenames,
                             classes = classes):
            return ("halo")
        else:
            return ("fail")        
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
        config = {"/": {},
                  "/post": {}
        }
        cherrypy.quickstart(Post(), config=config)
        print ("model shape", pd.cached_model.shape)
 
