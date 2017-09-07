__author__ = "Huangxuanyu Gong"

### Necessary imports.
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

### Make a News Data Frame. 
newsDF = pd.DataFrame()

### Sports Data
## Read sports and clean files
for sportsNews in os.listdir('./train/sports'):
    openSportsNews = open('./train/sports/' + sportsNews, 'r', encoding = 'latin1')
    readSportsNews = openSportsNews.readlines()
    openSportsNews.close()
 
    # Remove header
    delete = []
    for innerString in readSportsNews:
        if innerString is not '\n':
            delete.append(innerString)
        else: 
            break   
    for dele in delete:
        readSportsNews.remove(dele)

    # Remove '\t', '_', '|', '~' showed in string 
    deleteSymbol = []
    for i in range(len(readSportsNews)):
        if '\t' in readSportsNews[i]:
            readSportsNews[i] = readSportsNews[i].replace('\t','')
        if ('_' in readSportsNews[i]) or ('|' in readSportsNews[i]) or ('~' in readSportsNews[i]):
            deleteSymbol.append(readSportsNews[i]) 
    for delS in deleteSymbol:
        readSportsNews.remove(delS)
    
    # Make the sports news list as a sports new string
    sportsNewsString = ""
    for innerString in readSportsNews:
        sportsNewsString += innerString

    # Append to the newDf
    newsDF = newsDF.append({"News": sportsNewsString, "Type": 'Sports'}, ignore_index = True)


### Politics Data
## Read politics and clean files (Data is pretty clean, no need to clean data)
for politicsNews in os.listdir('./train/politics'):
    openPoliticsNews = open('./train/politics/' + politicsNews, 'r')
    readPoliticsNews = openPoliticsNews.read()
    openPoliticsNews.close()

    # Append to the newDF
    newsDF = newsDF.append({"News": readPoliticsNews, "Type": 'Politics'}, ignore_index = True)
    

# Acutal analysis part.
cv = CountVectorizer()
cv_fit = cv.fit_transform(newsDF['News'].values)

classifier = MultinomialNB()
targets = newsDF['Type'].values
classifier.fit(cv_fit, targets)

### Testing part. 
## The first test a sports news:
testfile1 = open('./test/sports/test1.txt', 'r')
file1 = [testfile1.read()]
print(file1)
example1 = cv.transform(file1)
prediction1 = classifier.predict(example1)
print(prediction1)

## The second test a politics news:
testfile2 = open('./test/politics/00765.txt', 'r')
file2 = [testfile2.read()]
print(file2)
example2 = cv.transform(file2)
prediction2 = classifier.predict(example2)
print(prediction2)

## The third test a pilitics news:
testfile3 = open('./test/politics/00035.txt', 'r')
file3 = [testfile3.read()]
print(file3)
example3 = cv.transform(file3)
prediction3= classifier.predict(example3)
print(prediction3)
