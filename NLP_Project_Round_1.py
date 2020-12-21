

#    _   _  _       _____       __          __              _     _____  _                    _              _____    ____    _____ 
#   | \ | || |     |  __ \      \ \        / /             | |   / ____|| |                  | |    ___     |  __ \  / __ \  / ____|
#   |  \| || |     | |__) |______\ \  /\  / /___   _ __  __| |  | |     | |  ___   _   _   __| |   ( _ )    | |__) || |  | || (___  
#   | . ` || |     |  ___/|______|\ \/  \/ // _ \ | '__|/ _` |  | |     | | / _ \ | | | | / _` |   / _ \/\  |  ___/ | |  | | \___ \ 
#   | |\  || |____ | |             \  /\  /| (_) || |  | (_| |  | |____ | || (_) || |_| || (_| |  | (_>  <  | |     | |__| | ____) |
#   |_| \_||______||_|              \/  \/  \___/ |_|   \__,_|   \_____||_| \___/  \__,_| \__,_|   \___/\/  |_|      \____/ |_____/ 
#                                                                                                                                   
#

# IMPORTING LIBRARIES                                                                                                                                   
import wordcloud
import pandas as pd
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import textblob
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Importing Text Files for constructing Corpus
file1=open("the_hairy_ones.txt","r");
file2=open("the_happy_castaway.txt","r");
t1=file1.read()
t2=file2.read()
file1.close()
file2.close()

# Functions for Tokenization and Word Cloud
def calculate_frequencies(file_contents):
    punctuations = '''!()-[]{};:'"<>./?@#$%^&*_~'''
    result = {}
    a = file_contents.split()
    for word in a:
            for letter in word:
                if letter in punctuations:
                    letter.replace(punctuations,"")  
            if word.lower() not in result.keys():
                result[word.lower()]=1
            else:
                result[word.lower()]+=1
     
    return constructingWordCloud(result)

def calculate_frequencies_removing_stopwords(file_contents):
    punctuations = '''!()-[]{};:'"<>./?@#$%^&*_~'''
    stop_words = set(stopwords.words('english'))  
    uninteresting_words = ["project","gutenberg-tm","gutenberg","the", "a", "to", "if", "is", "in","into","it", "of", "and", "or","on", "an", "as", "i", "me", "my", \
    "we", "our", "ours", "you", "your", "yours", "he", "she", "him", "his", "her", "hers", "its", "they", "them", \
    "their", "what", "which", "who", "work","whom", "this", "that", "am", "are", "was", "were", "be", "been", "being", \
    "have", "has", "had", "do", "does", "did", "but", "at", "by", "with", "from", "here", "when", "where", "how", \
    "all", "any", "for","*","both", "each", "few", "more", "some", "such", "no", "nor", "too", "very", "not", "can", "will", "just"]
    result = {}
    a = file_contents.split()
    for word in a:
        if word in stop_words:
            pass
        else:
            for letter in word:
                if letter in punctuations:
                    letter.replace(punctuations,"")
            if word.lower() not in uninteresting_words:
                if word.lower() not in result.keys():
                    result[word.lower()]=0
                else:
                    result[word.lower()]+=1
    return constructingWordCloud(result)

def tokenziation(file_contents):
    punctuations = '''!()-[]{};:'"<>./?@#$%^&*_~'''
    stop_words = set(stopwords.words('english'))  

    result = {}
    a = file_contents.split()
    for word in a:
        if word in stop_words:
            pass
        else:
            for letter in word:
                if letter in punctuations:
                    letter.replace(punctuations,"")
            if word.lower() not in result.keys():
                result[word.lower()]=1
            else:
                result[word.lower()]+=1
    return result


def length_Vs_Freq_Histogram(file_contents,title):
    results=tokenziation(file_contents)
    hist={}
    for word in results:
            if len(word) not in hist.keys():
                hist[len(word)]=1    
            else:
                hist[len(word)]+=1
    
          
    plt.bar(list(hist.keys()), hist.values(), color='r')
    plt.xlabel("Word Length") 
    plt.ylabel("Frequency") 
    plt.title('Histogram of Word Length and Frequency : '+title) 
    plt.show()            
    
def constructingWordCloud(result):
    cloud = wordcloud.WordCloud()
    cloud.generate_from_frequencies(result)
    return cloud.to_array()

def displayWordcloud(image,titleDescription):
    plt.imshow(image, interpolation = 'nearest')
    plt.title(titleDescription)
    plt.axis('off')
    plt.show()
    
# Display your wordcloud image
image_t1 = calculate_frequencies(t1)
image_t2 = calculate_frequencies(t2)
image_t1_removing_stopwords = calculate_frequencies_removing_stopwords(t1)
image_t2_removing_stopwords = calculate_frequencies_removing_stopwords(t2)
displayWordcloud(image_t1,"Word Cloud for T1")
displayWordcloud(image_t2,"Word Cloud for T2")
displayWordcloud(image_t1_removing_stopwords,"Word Cloud removing Stopwords for T1")
displayWordcloud(image_t2_removing_stopwords,"Word Cloud removing Stopwords for T2")
length_Vs_Freq_Histogram(t1,"T1")
length_Vs_Freq_Histogram(t2,"T2")
 


#POS Tagging
from collections import OrderedDict 
import operator
sample=tokenziation(t1)
model=nltk.pos_tag(sample)
graph={}
for i,tag in model:
        if tag not in graph.keys():
            graph[tag]=1    
        else:
            graph[tag]+=1
graph=dict(sorted(graph.items(), key=operator.itemgetter(1),reverse=True)) 
c=0
top_10_tags={}
for i in graph:
    if(c==10):
        break
    else:
        top_10_tags[i]=graph[i]
        c+=1
    
    
plt.bar(list(top_10_tags.keys()), top_10_tags.values(), color='g')
plt.xlabel("POS Tag") 
plt.ylabel("Count") 
plt.title('Histogram of Top 10 POS Tag and their count for T1 : ') 
plt.show()    