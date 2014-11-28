import mailbox
import os
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

                if (payloadstr!=None 
                    and (len(payloadstr)>200)):

                    counter+=1
                    # Quick'n'dirty parsing tricks
                    msgStr+=payloadstr 
                    msgStr=msgStr.replace('=\r\n', '')
                    msgStr=msgStr.replace('<br>','\n')
                    fname = ("email_"+str(counter)+'_'
                             +baseparse(msg['Subject'])
                             +".txt") 
                    path = os.path.join(output_path,
                                        fname)
                    f=open(path, 'w')
                    f.write(st.strip(msgStr))
                    f.close()

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
        msgBody = payload[:200]
    return msgBody


all_labels = ['Trash', 'Important', 'Inbox', 'Chat', 'Starred', 'Unread', 'Sent']
unread_labels = ['Unread']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","-m","--mbox","--input", dest="input",
                        help="input file path, mbox file")
    parser.add_argument("-o", "--output", dest = "output",
                        help="output file path")
    parser.add_argument("-t","-c","-type","-categories", 
                        "-l", "-labels", dest="labels", 
                        type=str, nargs="*", default=all_labels,
                        help="types of mails to extract - Trash, Important, Inbox, Chat, Starred, Unread, Sent")
    args = parser.parse_args()

    if not (args.input or args.output):
        parser.error("Required: both --input and --output paths")
    else:
        args.input  = os.path.realpath(args.input)
        args.output = os.path.realpath(args.output)
        if not(os.path.exists(args.input)):
            parser.error("input path doesn't exist")
        if not(os.path.exists(args.output)):
            parser.error("output path doesn't exist")
        lc_labels = map(lambda t: t.lower(), all_labels)
        for label in args.labels:
            if label.lower() not in lc_labels:
                parser.error("Invalid label, "+label+" please check help for for valid labels")
    showMbox(args.input, args.output, args.labels)
