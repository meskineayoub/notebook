import tweepy
import re
import pickle
from tweepy import OAuthHandler


#Initialising the keys
consumer_key='A7zEJHYcutt3U37c4JEEcSXtA'
consumer_secret='ZCseQI6Tvo604hLNt2nXwpTLKqtx8bQwXgpbkeAVnodyoanORw'
access_token='1107606361901289472-1g12pcv7I03dzzPAvqVBNqtUidDItP'
access_secret='OMEha5VviMEmf8mBSDNhGjh17Z6qXtd8Ea6sCPtdboLOl'



auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)



args = ['covide']
api = tweepy.API(auth,timeout=10)

list_tweets = []

query = args[0]

if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='re').items(100):
        list_tweets.append(status.text)
      
with open('tfidfmodel.pickle','rb') as f:
    vectorizer = pickle.load(f)
with open('classifierB.pickle','rb') as f:
    clfB = pickle.load(f)
with open('classifierL.pickle','rb') as f:
    clfL = pickle.load(f)
    
total_posB = 0
total_posL= 0
total_negL=0
total_negB = 0

for tweet in list_tweets:
    tweetOr=tweet
    tweet=re.sub(r"^https://t.co/[a-zA-z0-9]*"," ",tweet)
    tweet=re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet=re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet=tweet.lower()
    tweet=re.sub(r"that's","that is",tweet)
    tweet=re.sub(r"there's","there is",tweet)
    tweet=re.sub(r"what's","what is",tweet)
    tweet=re.sub(r"where's","where is",tweet)
    tweet=re.sub(r"it's","it is",tweet)
    tweet=re.sub(r"who's","who is ",tweet)
    tweet=re.sub(r"i'm","i am",tweet)
    tweet=re.sub(r"she's","she is",tweet)
    tweet=re.sub(r"they're","they are",tweet)
    tweet=re.sub(r"ain't","am not",tweet)
    tweet=re.sub(r"wouldn't","would not",tweet)
    tweet=re.sub(r"shouldn't","should not",tweet)
    tweet=re.sub(r"can't","can not",tweet)
    tweet=re.sub(r"couldn't","could not",tweet)
    tweet=re.sub(r"won't","will not",tweet)
    tweet=re.sub(r"\W"," ",tweet)
    tweet=re.sub(r"\d"," ",tweet)
    tweet=re.sub(r"s+[a-z]\s+"," ",tweet)
    tweet=re.sub(r"s+[a-z]\$"," ",tweet)
    tweet=re.sub(r"^[a-z]\s+"," ",tweet)
    tweet=re.sub(r"\s+"," ",tweet)
    sentB = clfB.predict(vectorizer.transform([tweet]).toarray())
    sentL = clfL.predict(vectorizer.transform([tweet]).toarray())
    if sentB[0] == 1:
        total_posB+=1    
    else:
        total_negB+=1
        
    print("\n<<<<<:::::::Bayes polarity :::::::::>>>> ",sentB[0],'\n',tweetOr,'\n\n',"<<<<<<<:::::::Logistic polarity::::::::::::>>>> ",sentL[0],'\n',tweetOr)
    if sentL[0] == 1:
        total_posL+=1    
    else:
        total_negL+=1

#Plottng the bar chart
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive_B','Positive_L','Negative_B','Negative_L']
y_pos = np.arange(len(objects))
plt.ylabel('Number')
plt.title('Number of Positive and Negative Tweets using base algorithm and logistic algorithm')

plt.bar(y_pos,[total_posB,total_posL,total_negB,total_negL],alpha=0.5,color=['blue','green','red','black'])
plt.xticks(y_pos,objects)

plt.show()
"""
plt.bar(y_pos,[total_posL,total_negL],alpha=0.5,color=['green','red'])
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Positive and Negative Tweets using Logistic regression algorithm')

plt.show()
"""

labels = 'negative_B', 'positive_B', 'negative_L', 'positive_L'
sizes = [total_negB, total_posB, total_negL, total_posL]
explode = (0.1, 0.1, 0.2, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()

     
