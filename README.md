# Sports and Politics News Classifier

### Introduction:
--- 
Giving a training set of sports and politics news articles, how can a machine classify the news types for other articles? This project uses Naive Bayes algorithm to predict the news types.  

### Source:
---
RSS feed from CNN, Fox News, and NBC News from Nov. 2016 to Jul. 2017 are used as training and testing set. 

### Data Processing:
---
1. Read sports and politics news from 'train' repository, and made a Pandas DataFrame containing all the news and its category.
2. Applied CountVectorizer to find term frequency - inverse document frequency (tf-idf) for all the news.  
3. Utilized Naive Bayes algorithm with MultinomialNB to train and predict the news articles categories. 

### Result and Conclusion:
---
Results:<br>
##### First article: <br>
['As the emerging face of the NHL, Edmonton Oilers center Connor McDavid seemingly has it all: unparalleled speed, a Hart Trophy as MVP, and a supporting cast that makes his team a legitimate title contender this coming season.\n']<br>
Project result:<br>
['Sports']
<br><br>

##### Second article:<br>
["Phil Roe's scheduling conflict: 'I'm going to my wedding Saturday'\nThe buzz on Capitol Hill Wednesday afternoon about a possible health care vote had lawmakers wondering if they would be able to leave town on Thursday for recess."]<br>
Project result:<br>
['Politics']<br>
<br>

##### Third article:<br>
["Schumer faces failure in first test as leader\nWill Senate Dems be successful in blocking Gorsuch's confirmation?\n"]<br>
Project result:<br>
['Politics']<br>

Conclusion:<br>
As the results show, Naive Bayes algorithm successfully predict the categories of testing news articles. 



