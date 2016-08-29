import os 
import xml.etree.ElementTree

train_address= "train"
test_address="test"

train_neg_address= train_address+"/neg"
train_pos_address= train_address+ "/pos"
test_neg_address= test_address+"/neg"
test_pos_address= test_address+ "/pos"
train_unsup_address= train_address+ "/unsup"


adresses = [train_neg_address,train_pos_address,test_neg_address,test_pos_address, train_unsup_address]
for add in adresses:
	if not os.path.exists(add):
		os.makedirs(add)

counter_neg = 0
counter_pos = 0
for xmlfile in ['books/train.review', 'dvd/train.review', 'music/train.review']:
	e = xml.etree.ElementTree.parse(xmlfile).getroot()

	print xmlfile
	for atype in e.findall('item'):
		
		if(atype.find('rating').text=="2.0" or atype.find('rating').text=="1.0"):
			counter_neg+=1
			f = open(train_neg_address+"/"+str(counter_neg)+"_"+atype.find('rating').text+".txt","w")
			f.write(atype.find('text').text.encode('utf8'))
			f.close()
		if(atype.find('rating').text=="4.0" or atype.find('rating').text=="5.0"):
			counter_pos+=1
			f = open(train_pos_address+"/"+str(counter_pos)+"_"+atype.find('rating').text+".txt","w")
			f.write(atype.find('text').text.encode('utf8'))
			f.close()	
counter_neg = 0
counter_pos = 0
for xmlfile in ['books/test.review','dvd/test.review','music/test.review']:
	e = xml.etree.ElementTree.parse(xmlfile).getroot()

	print xmlfile
	for atype in e.findall('item'):
		
		if(atype.find('rating').text=="2.0" or atype.find('rating').text=="1.0"):
			counter_neg+=1
			f = open(test_neg_address+"/"+str(counter_neg)+"_"+atype.find('rating').text+".txt","w")
			f.write(atype.find('text').text.encode('utf8'))
			f.close()
		if(atype.find('rating').text=="4.0" or atype.find('rating').text=="5.0"):
			counter_pos+=1
			f = open(test_pos_address+"/"+str(counter_pos)+"_"+atype.find('rating').text+".txt","w")
			f.write(atype.find('text').text.encode('utf8'))
			f.close()	

counter=0
for xmlfile in ['books/unlabeled.review', 'dvd/unlabeled.review', 'music/unlabeled.review']:
	e = xml.etree.ElementTree.parse(xmlfile).getroot()

	print xmlfile
	for atype in e.findall('item'):
		counter+=1
		try:
			f = open(train_unsup_address+"/"+str(counter)+"_"+atype.find('rating').text+".txt","w")
			f.write(atype.find('text').text.encode('utf8'))
		except:
			print "error happend"	
		else:	
			f.close()
			print counter
print counter