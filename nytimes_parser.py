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

if __name__ == "__main__":
    process_files(test_dir, output_dir)
    print "Execution complete"
