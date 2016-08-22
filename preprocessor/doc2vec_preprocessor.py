import glob
import os

# random shuffle
from random import shuffle
# gensim modules
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import sys

import logging
log = logging.getLogger();
log.setLevel=(logging.DEBUG)

dataset_path='./aclImdb/'


def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), ["TRAIN_POS" + '_%s' % item_no]))
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]#-1 means reverse the sort

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict




class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            currdir = os.getcwd()
            os.chdir(source)
            for item_no,ff in enumerate(glob.glob("*.txt")):
                with open(ff, 'r') as f:
                    self.sentences.append(TaggedDocument(f.readline().strip().split(), [prefix + '_%s' % item_no]))
            print item_no, prefix        
            os.chdir(currdir)
            # with utils.smart_open(source) as fin:
            #     for item_no, line in enumerate(fin):
            #         self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences
        



def create_or_load_doc2vec_model(filename):
    if os.path.isfile(filename):
        model = Doc2Vec.load(filename)
        return model
    else:
    	log.info('source load')
    	sources = {'aclImdb/test/neg/':'TEST_NEG',
                   'aclImdb/test/pos/':'TEST_POS',
                   'aclImdb/train/neg/':'TRAIN_NEG',
                   'aclImdb/train/pos/':'TRAIN_POS'}

    	log.info('TaggedDocument')
    	sentences = TaggedLineSentence(sources)

        # print sentences.to_array()
    	model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    	model.build_vocab(sentences.to_array())
    	log.info('Epoch')
    	for epoch in range(10):
    		log.info('EPOCH: {}'.format(epoch))
    		model.train(sentences.sentences_perm())

    	log.info('Model Save')
    	model.save('./imdb.d2v')
    	return model



# loading dov2vec model
model = create_or_load_doc2vec_model('./imdb.d2v')
    


