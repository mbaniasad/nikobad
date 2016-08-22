# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tensorflow as tf
class config:
    
    def __init__(self,dataset_path,number_of_words_used_in_embedding,dropout):
        self.dataset_path = dataset_path
        self.number_of_words_used_in_embedding  = number_of_words_used_in_embedding
        self.dropout = dropout
    def setting_name(self):
        return 'lstm'+'ds-'+self.dataset_path +'embedding-'+str(self.number_of_words_used_in_embedding)+'dropout-'+str(self.dropout)


lstm_configs = [config( 'imdb_doc2wec_sentiment.pkl', 10000, 0.1),
                config( 'imdb_doc2wec_sentiment.pkl', 10000, 0.1),
                config( 'imdb_doc2wec_sentiment.pkl', 10000, 0.1),  
                config( 'imdb_doc2wec_sentiment.pkl', 10000, 0.1),  
                config( 'imdb_doc2wec_sentiment.pkl', 10000, 0.1)  ]

for cl in lstm_configs:
    with tf.Graph().as_default():
        print("running lstm_on_"+cl.setting_name())
        # IMDB Dataset loading
        train, test, _ = imdb.load_data(path=cl.dataset_path, n_words=cl.number_of_words_used_in_embedding,
                                        valid_portion=0.1)
        trainX, trainY = train
        testX, testY = test

        # Data preprocessing
        # Sequence padding
        trainX = pad_sequences(trainX, maxlen=100, value=0.)
        testX = pad_sequences(testX, maxlen=100, value=0.)
        # Converting labels to binary vectors
        trainY = to_categorical(trainY, nb_classes=2)
        testY = to_categorical(testY, nb_classes=2)

        # Network building
        net = tflearn.input_data([None, 100])
        net = tflearn.embedding(net, input_dim=cl.number_of_words_used_in_embedding, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=cl.dropout)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')

        # Training
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/'+cl.setting_name())
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
                  batch_size=32, n_epoch=10)

