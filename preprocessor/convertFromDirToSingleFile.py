import os
import glob

def readDatasFromDir(DirAddress):
    sentences = []
    currdir = os.getcwd()
    os.chdir(DirAddress)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    return sentences

datasetAddress = "./aclImdb/"
trainPositive = [datasetAddress + "train/pos/", "train-pos.txt"]
trainNegitive = [datasetAddress + "train/neg/", "train-neg.txt"]
testPositive = [datasetAddress + "test/pos/" ,"test-pos.txt"]
testNegitive =[ datasetAddress + "test/neg/","test-neg.txt"]

addresses =[ trainPositive, trainNegitive, testPositive, testNegitive]
for directoryAddress,filename in addresses:
    with open (filename,'w') as destinationFile:
        for line in readDatasFromDir(directoryAddress):
            destinationFile.write(line)
