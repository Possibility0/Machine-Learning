#Sentiment analysis
import nltk
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
stopwords = stopwords.words('english')

#Tokenizing words for nltk format
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

def classify(positive,negative):
    
    #opening & reading the files
    pos_file = open(positive)
    neg_file = open(negative)
    text = pos_file.read()
    text2 = neg_file.read()

    #Cleaning the data by removing stopwords
    clean_pos = [word for word in text.split() if word not in stopwords]
    clean_neg = [word for word in text2.split() if word not in stopwords]
    
    #making a list of positive & negative words
    pos = []
    for i in clean_pos:
        pos.append([format_sentence(i),'pos'])

    neg = []
    for i in clean_neg:
        neg.append([format_sentence(i),'neg'])

    #Creating training & test set
    training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))] 
    test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

    #Building a classifier
    classifier = NaiveBayesClassifier.train(training)

    #Making predictions
    print('negative tweets classified by the classifier as',':',classifier.classify(format_sentence(str(neg))))
    print('positive tweets classified by the classifier as',':',classifier.classify(format_sentence(str(pos))))
    

    
   

    

    

   

     
    

    


