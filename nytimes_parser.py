"""
Script to parse XML from the NYT corpus 
:-: Anjishnu Kumar

Simple usage:

:-: python nytimes_parser.py input_dir output_dir

"""

import sys
from xml.dom import minidom
import os
import xml.etree.ElementTree as ET

print "NYTimes XML Parser loaded"
test_dir = "/home/user/Huffpost/LDA_demo/NYTimes/corpus/corpus/data/2006/12/12"
output_dir = "/home/user/Huffpost/LDA_demo/nytimes_data"

def gen_file_paths(inp_dir):
    for fname in os.listdir(inp_dir):
        yield fname.split('.')[0], os.path.join(inp_dir, fname)

def strip_xml (xml_string):
    tree = ET.fromstring(xml_string)
    no_tags = ET.tostring(tree, encoding= 'utf8', method='text')
    return no_tags

def load_file(fpath):
    xmldoc = minidom.parse(fpath)
    blocklist = xmldoc.getElementsByTagName('block')
    titles    = xmldoc.getElementsByTagName('title')
    title = strip_xml(titles[0].toxml())
    for item in blocklist:
        #print item.attributes['class']
        if item.attributes['class'].value=="full_text":
            text = strip_xml(item.toxml())
    return title, text

def store(store_path, text):
    with open(store_path, 'w') as f:
        f.write(text)
    return True


"""
Currently just working with extracted directories
"""
def process_files(inp_dir, out_dir):
    for name, full_path in gen_file_paths(inp_dir):
        try:
            title, text = load_file(full_path)
            title = title.replace(","," ")
            title+=".txt"
            title=title.replace(" ", "_")
            out_path = os.path.join(out_dir,title)
            store(out_path,text)
        except Exception as e:
            print "Couldn't process", name
            print e.message

"""

"""
def main():
    args = sys.argv[1:]
    if len(args)!=2:
        print "Invalid number of arguments"
        print "usage->"
        print ":-: python nytimes_parser input_dir output_dir"
        print "NOTE: remember to extract the tgz model in the directory you are extracting from"
        return 
    input_dir, output_dir = args
    input_dir, output_dir = map(os.path.abspath, [input_dir, output_dir])
    print input_dir, output_dir
    #process_files(input_dir, output_dir)

if __name__ == "__main__":
    #process_files(test_dir, output_dir)
    main()
    print "Execution complete"
