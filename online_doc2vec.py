# -*- coding: utf-8 -*-
import tflearn
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt

net = tflearn.input_data([None, 100])
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='Momentum', learning_rate=0.001, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tflearn_logs/')
model.load("doc2vecds-germanAmazonReviews.pkloptimizer-Momentumloss-categorical_crossentropy")

gensimModel = Doc2Vec.load("germanAmazonReveiews.d2v")


def doc2vec_online_prediction(sentence):
	global gensimModel, model
	sentence_array = sentence.split()
	vectors = []
	for i in range(len(sentence_array)):
		sentence = sentence_array[0:i+1]
		print sentence
		vector= gensimModel.infer_vector(sentence, alpha=0.1, min_alpha=0.0001, steps=5)
	 	vectors.append(vector)
	predictions = model.predict(vectors)
	return predictions
		



predictions= doc2vec_online_prediction("Also erstmal fange ich mit der Verpackung an. Die Kopfhöhrer befinden sich in einer Plastikverpackung, welche keinerlei Öffnungsmöglichkeiten bietet, ohne Hilfsmittel. Selbst mit einer Schere hat man nur schwer eine Chance, ich habe mich also mit Schere und Cuttermesser an die Verpackung gewagt. Ich habe gut 5 Minuten gebraucht, bis das Ding 'offen' war. Dabei musste man einerseits darauf achten, sich nicht an dem scharfen Plastik zu schneiden und andererseits musste man aufpassen, die Kabel nicht zu verletzten. Allein für die Verpackung gibt es einen Stern Abzug.  Nun zu den Köpfhöhrern selbst. Bin total entäuscht, statt den 'enormen' Bass zu hören, welcher hier immer wieder in Rezessionen erwähnt wird, höre ich mehr oder weniger nur etwas blechiges raus, aber keinen wirklichen Bass. Wenn ich bei meinem Turtle Beach Headset den Bass am Rädchen auf 0 drehe, dann erhalte ich die gleiche Bass-Intensität wie bei diesen hier. Keine Ahnung wie man hier solche Aussagen wie 'für Bass-Fetischisten' machen kann.  Der Sound ist also nicht gerade überragend (meiner Ansicht nach etwas blechern). Bass ist kaum bis gar nicht vorhanden. Es gibt hier dutzende Alternativen für die Hälfte des Preises, welche sicherlich mindestens die gleiche Qualität, wenn nicht sogar bessere bieten.  Ich schreibe selten Rezessionen, aber hier fühle ich mich fast schon dazu verpflichtet, vor dem Produkt zu warnen. Ich bin seit über 8 Jahren Amazon Kunde und habe in der Zeit noch nie etwas zurück gehen lassen. Aber bei diesem Produkt ziehe ich es tatsächlich in erwägung, nicht weil mir die 20€ wehtun würden, nein, einfach weil ich es nicht einesehen möchte, das hier mit 2€ Kopfhöhrern Kohle gemacht wird.")
print predictions




plt.plot(predictions)
plt.ylabel('some numbers')
plt.show()
