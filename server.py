import web
import json
import process_data as pd
from argparse import ArgumentParser

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
        i= web.input()
        text = web.data().encode('utf8')     
        json_obj = json.loads(text)
        filenames = clean_json(json_obj)
        for name in filenames:
            print name
        if pd.subset_run(filenames):
            return "halo"
        else:        return "fail"

    def OPTIONS(self):
        #i =  web.input()
        #print type(i), dir(i)
        return

def clean_json(dirty_json):
    dirty_json = dirty_json[0]
    name_of = lambda x: x['__data__']['Name']
    return map(name_of, dirty_json)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r","--run", dest="run",
                        action = "store_true",
                        help="kick off computation")
    args = parser.parse_args()
    if args.run:
        pd.compose(pd.global_builders, pd.global_sizes)
    app = web.application(urls, globals())
    app.run()
