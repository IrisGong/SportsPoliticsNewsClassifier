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
## Read the sports news from test folder
for testSnews, testPnews in zip(os.listdir('./test/sports'), os.listdir('./test/politics')):
    openTSportsNews = open('./test/sports/' + testSnews, 'r', encoding = 'latin1')
    readTSportsNews = openTSportsNews.read()
    openTSportsNews.close()
    openTPoliticsNews = open('./test/politics/' + testPnews, 'r')
    readTPoliticsNews = openTPoliticsNews.read()
    openTPoliticsNews.close()
    
    test_example = [readTSportsNews, readTPoliticsNews]
    example_counts = cv.transform(test_example)

    predictions = classifier.predict(example_counts)

    print('Expecting: Sports, Politics')
    print('Actual result: ')
    print(predictions)
    
