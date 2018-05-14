import nltk
import os
print(os.getcwd())
os.chdir('/media/achint/INSOFE/Movie Review_Sentiment Lab')
import random
from sklearn.metrics import confusion_matrix
#opening text
print(os.listdir(os.getcwd()))
with open('positive.txt','r',encoding='utf-8',errors='ignore') as p:
	short_pos=p.read()
print(short_pos)
with open('negative.txt','r',encoding='utf-8',errors='ignore') as n:
	short_neg=n.read()
print(short_neg)

short_neg=short_neg.lower()
short_pos=short_pos.lower()

posidocuments=short_pos.split('\n')
print(posidocuments[0])
negdocuments=short_neg.split('\n')
print(negdocuments[0])

#takiing first 1000
posdocs=posidocuments[:1000]
negdocs=negdocuments[:1000]
print(posdocs[999])
print(negdocs[999])

#adding p and n to positive and negative
documents=[]
for p in posdocs:
	documents.append((p,'p'))
for n in negdocs:
	documents.append((n,'n'))
print(documents)

#tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer(r'\w+')

#stopwords
from nltk.corpus import stopwords
stopwords=stopwords.words('english')

#tagging wrt type of word(noun,adjective etc.)
tokens=tokenizer.tokenize(posdocs[0])
print(nltk.pos_tag(tokens))

#taking only adjectives
allowed_word_type=['JJ','JJR','JJS']
all_words=[]
print(all_words)
for doc,label in documents:
	words=tokenizer.tokenize(doc)
	tagged_words=nltk.pos_tag(words)
	for word,tag in tagged_words:
		if tag in allowed_word_type:
			all_words.append(word)
print(all_words)
print(len(all_words))

#this list not unique, so do frequency distribution
freq_dist=nltk.FreqDist(all_words)
print(all_words[10])
print(freq_dist)
print(freq_dist.most_common(10))

#set function makes set of unique values
word_features=set(all_words)
print(len(word_features))

#loop to iterate on each sentence to find presence of word
def find_features(documents):
	document_tokens=tokenizer.tokenize(documents)
	features={}
	for w in word_features:
		features[w]=(w in document_tokens and w not in stopwords)
	return features
featuresets=[(find_features(rev),category) for (rev,category) in documents]
print(featuresets[0])

#randomly making test and train sets
random.shuffle(featuresets) 
training_set=featuresets[:1800]
test_set=featuresets[1800:]

#naive bayes on sets
classifier=nltk.NaiveBayesClassifier.train(training_set)
train_acc=nltk.classify.accuracy(classifier,training_set)*100
print('Train Accuracy',train_acc)
#shows the most info of naive bayes
classifier.show_most_informative_features(20)

#testing the model
predictions=classifier.classify_many(x[0] for x in test_set)
print(confusion_matrix(predictions,[x[1] for x in test_set]))
test_accuracy=nltk.classify.accuracy(classifier,test_set)
print(test_accuracy)
