import tweepy
import pandas as pd
import csv
import sys
import time
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

#Initialized for Twitter API usage
access_key = "1155797393360560129-SBwjZHMka43b2xjkz8e2gtD5401rPh"
access_secret = "c730hlpddNFUDPgvwdNNt6OCorWj6cvoXYZVk4AdrUK3J"
consumer_key = "0CTwWRbopMM3WLUbS5HqOQ3zM"
consumer_secret = "QvJ2seNi7U0W7cNQCLji3gQ5jK7WWiwtcPjP9SdEHBcHN8bZqw"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


filename = 'movies.csv'
df = pd.read_csv(filename)
print(df)
user = list(df.user)

#Filtered Tweets from links, non ascii characters, number embedded in words, stopwords
def filter_tweets(filter_list):
    for i in range(len(filter_list)):
        filter_list[i] = re.sub(r'https://[A-Za-z0-9].co/[A-Za-z0-9]+', '', filter_list[i])
        filter_list[i] = re.sub(r'http://[A-Za-z0-9].co/[A-Za-z0-9]+', '', filter_list[i])
        filter_list[i] = re.sub(r'[^\x00-\x7f]', r'', filter_list[i])
        filter_list[i] = re.sub(r'@[A-Za-z0-9]+', '', filter_list[i])
        #filter_list[i] = re.sub(r'[b"]', '', filter_list[i])

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(filter_list[i])
        tweet_lst = []

        for x in word_tokens:
            if x not in stop_words:
                tweet_lst.append(x)

        filter_list[i] = " ".join(tweet_lst)

    return filter_list

#Downloaded Tweets through API
cnt = 0
last_one = ""
error_lst = []
for i in user:
    name = i
    tweet_texts = []
    alltweets = []

    try:
        new_tweets = api.user_timeline(screen_name=name, count=200)
        for i in new_tweets:
            tweet_texts.append(i.text)

        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        while len(new_tweets) > 0:
          
            new_tweets = api.user_timeline(screen_name=name, count=200, max_id=oldest)
            for i in new_tweets:
                tweet_texts.append(i.text)
            alltweets.extend(new_tweets)
            oldest = alltweets[-1].id - 1
            #print("...%s tweets downloaded so far" % (len(alltweets)))


        tweet_texts = filter_tweets(tweet_texts)
        outtweets = [i.encode("utf-8") for i in tweet_texts]
        #print(outtweets)
        print(name)
        dataframe = pd.DataFrame(outtweets, columns=['tweets'])
        to_csv = dataframe.to_csv('Movie_Tweets/' + name+ '.csv', index=None, header=True)
        cnt += 1
        print("count: " + str(cnt)+ "-->" + name)

    except:
        error_lst.append(name)



print(str(cnt) + " done...")

print("Errors: ", error_lst)
