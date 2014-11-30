import mailbox
import os
import time
from HTMLParser import HTMLParser
import string
from argparse import ArgumentParser

MBOX_PATH = os.path.join(os.getcwd(), 'all.mbox')
outpath = os.path.join(os.getcwd(), 'email_data')

class HTMLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self,data):
        self.fed.append(data)

    def get_data(self):
        return "".join(self.fed)

    def strip(self, data):
        self.feed(data)
        return self.get_data()

def baseparse(inp_string):
    out = []
    for letter in inp_string:
        if letter not in string.punctuation:
            out+=[letter]
    return "".join(out).replace(" ", "_")

def matchLabels(labels, msg):
    if msg['X-Gmail-Labels']:
        msg_labels = msg['X-Gmail-Labels'].split(',')
        #print msg_labels
        for mlabel in msg_labels:
            if mlabel in labels:
                return True
    return False

def showMbox(mbox_path, output_path=outpath, labels=None):
    print "Loading mbox data"
    box = mailbox.mbox(mbox_path)
    counter = 0

    for msg in box:
        if matchLabels(labels, msg):

            st = HTMLStripper()
            msgStr = ""
            if (msg['Subject']!=None 
                and len(msg['Subject'])!=0):           

                print "Message Number", counter, msg['Subject']
                msgStr+=msg['Subject']
                payloadstr = showPayload(msg)

                if (payloadstr!=None):
                    msgStr += parse_email(payloadstr)
                    counter+=1
                    # Quick'n'dirty parsing tricks
                    fname = ("email_"+str(counter)+'_'
                             +baseparse(msg['Subject'])
                             +".txt") 
                    path = os.path.join(output_path,
                                        fname)
                    
                    try:
                        f=open(path, 'w')
                        msgStr=st.strip(msgStr)
                        f.write(msgStr)
                        f.close()
                    except:
                        print "failed to strip the string"
                        print msgStr

def parse_email(payload_str):
    msg_str=payload_str 
    msg_str=msg_str.replace('=\r\n', '\n')
    msg_str=msg_str.replace('\r\n\r', '\n')
    msg_str=msg_str.replace('<br>','\n')
    return msg_str

def showPayload(msg):
    payload = msg.get_payload()
    msgBody = ""
    if msg.is_multipart():
        div = ''
        for subMsg in payload:
            msgBody+=showPayload(subMsg)
            div = '------------------------------'
    else:
        #print msg.get_content_type()
        #Restriction size to small messages
        #msgBody = payload[:200]
        msgBody = payload
    return msgBody


all_labels = ['Trash', 'Important', 'Inbox', 'Chat', 'Starred', 'Unread', 'Sent']
unread_labels = ['Unread']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","-m","--mbox","--input", dest="input",
                        help="input file path, mbox file")
    parser.add_argument("-o", "--output", dest = "output",
                        help="output file path")
    parser.add_argument("-t","-type", "-l", "--labels", 
                        dest="labels", type=str, nargs="*", 
                        default=all_labels,
                        help="types of mails to extract - Trash, Important, Inbox, Chat, Starred, Unread, Sent")
    args = parser.parse_args()

    if not (args.input or args.output):
        parser.error("Required: both --input and --output paths")
    else:
        args.input  = os.path.realpath(args.input)
        args.output = os.path.realpath(args.output)
        if not(os.path.exists(args.input)):
            parser.error("input path doesn't exist")
        else:
            print "MBOX File:", args.input
        
        if not(os.path.isdir(args.output)):
            parser.error("output path doesn't exist")
            print args.output, "is not a directory"
            print "creating", args.output
            os.makedirs(args.output)
        else:
            print "Output Dir:", args.output

        lc_labels = map(lambda t: t.lower(), all_labels)
        verified_labels = []
        for label in args.labels:
            if label.lower() not in lc_labels:
                parser.error("Invalid label, "+label+" please check help for for valid labels")
            else:
                verified_labels+=[label]
        print "Labels", verified_labels

    ts = time.time()
    showMbox(args.input, args.output, verified_labels)
    te = time.time()
    #Printing for benchmarking purposes - 
    #Pretty sure this step can be sped up a lot
    print "Total time for completion", te-ts, "seconds"
