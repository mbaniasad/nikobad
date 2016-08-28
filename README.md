#### needed packages
tensorflow Version: 0.10.0rc0
tflearn Version: 0.2.2

## to see the results in tensor board
```
tensorboard --logdir=tflearn_logs
#(You can navigate to http://0.0.0.0:6006)
```

# preprocessing your own dataset
 to use your own dataset you have to make it form like this
```
aclImdb/
    ├── test
    │   ├── neg
    │   └── pos
    └── train
    │   ├── neg
    │   ├── pos
    └── unsup
```
and use
```
cd  preprocessor
python imdb_preprocess.py
```


# pkl ed datasets
1- aclImdb which is accessable with preprocessor code

2- doc2vec_sentiment datasets from [doc2vec sentiment repo](https://github.com/linanqiu/word2vec-sentiments) **have to change the format of raw charachter data if want to use preprocessor on this**

3- amazon reviews dataset from (...) **German**
