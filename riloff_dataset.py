import csv
import re
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

class dataset():
    
    def __init__(self):
        self.affective = dict( )
        
    #===============================================================================
    def __addPunctFeatures(self, tweet):
    
        #FEATURES
        question_mark_RE_str = '\?'
        exclamation_point_RE_str = '\!'
        interrobang_RE_str = '[\?\!]{2,}'
         
        if len(re.findall(r'%s' % exclamation_point_RE_str, tweet)) > 0:
            tweet += " PUNCxEXCLAMATION_POINT"
        if len(re.findall(r'%s' % question_mark_RE_str, tweet)) > 0:
            tweet += " PUNCxQUESTION_MARK"
        if len(re.findall(r'%s' % interrobang_RE_str, tweet)) > 0:
            tweet += " PUNCxINTERROBANG"
        if any([len(s) > 2 and str.isupper(s) for s in tweet.split(" ")]):
            tweet += " PUNCxUPPERCASE"
        return tweet

    def __addEmoticonsFeatures(self, tweet):   
        emoticon_RE_str = '(?::|;|=|X|x)(?:-)?(?:\)|\(|D|P)'
         
        if len(re.findall(r'%s' % emoticon_RE_str, tweet)) > 0:
            tweet += " PUNCxEMOTICON"
        return tweet
    
    
    
    def __addAffectiveNormsFeature(self, tweet):   
        valence = 0
        arousal = 0
        dominance = 0
        counter = 0
        for word in tweet.split(' '):
            word = word.lower()
            if word in self.affective:
                valence += self.affective[word][0]
                arousal += self.affective[word][1]
                dominance += self.affective[word][2]
                counter += 1
                        
        if counter != 0:
            valence /= counter
            arousal /= counter
            dominance /= counter
                        
            if valence < 3:
                tweet += " LOW_VALENCE"
            elif valence < 7:
                tweet += " MEDIUM_VALENCE"
            else:
                tweet += " HIGH_VALENCE"
            if arousal < 3:
                tweet += " LOW_AROUSAL"
            elif arousal < 7:
                tweet += " MEDIUM_AROUSAL"
            else:
                tweet += " HIGH_AROUSAL"
            if dominance < 3:
                tweet += " LOW_DOMINANCE"
            elif dominance < 7:
                tweet += " MEDIUM_DOMINANCE"
            else:
                tweet += " HIGH_DOMINANCE"
        return tweet
    
    #===============================================================================
    def compute_comments_similarities(self, no_similarities):
        tweet_id_list = self.__get_tweet_ids()
        similarities = []
        for tweet_id in tweet_id_list:
            print(tweet_id)
            similarities.append(self.__comment_similarity(tweet_id, no_similarities))
    
    def __comment_similarity(self, tweet_id, no_similarities):
        conn = sqlite3.connect('riloff_dataset.db')
        c = conn.cursor()
        orig_tweet = list(c.execute("SELECT tweet_text FROM labeled_tweets WHERE id='%s'" % tweet_id))[0][0]
        c = conn.cursor()
        tweets = c.execute("SELECT tweet_text FROM past_author_tweets WHERE orig_tweet_id='%s' LIMIT %d" % (tweet_id, no_similarities))
        vect = TfidfVectorizer(min_df=1)
        tweets = list(tweets.fetchall())
        if len(tweets) == 0:
            print("No tweets!!")
            return []
        tfidf = vect.fit_transform([orig_tweet] + [t[0] for t in tweets])
        similarities = (tfidf * tfidf.T).A
        result = []
        for simi in similarities:
            result.append(simi[0])
            
        return result
    
    def __get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
    def __get_tweet_ids(self):
        conn = sqlite3.connect('riloff_dataset.db')
        cursor = conn.cursor()
        rows = cursor.execute(""" SELECT id FROM labeled_tweets """)
        return [row[0] for row in rows]
    
    def get_data(self, affective_norms=False, emoticons=False, punctuation=False):
        conn = sqlite3.connect('riloff_dataset.db')
        conn.text_factory = str
        cursor = conn.cursor()
        rows = cursor.execute(""" SELECT tweet_text, label FROM labeled_tweets """)
        
        if affective_norms:
            for row in csv.DictReader(open("Ratings_Warriner_et_al.csv")): 
                self.affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
        
        tweets_target = []
        all_tweets = []
        
        for tweet, label in rows:
            tweet = tweet.lower().replace("#sarcasm","").replace("#irony","").replace("#sarcastic","")
            if affective_norms:
                tweet = self.__addAffectiveNormsFeature(tweet)
            if punctuation:
                tweet = self.__addPunctFeatures(tweet)
            if emoticons:
                tweet = self.__addEmoticonsFeatures(tweet)
                
            if label == "SARCASM":
                tweets_target.append(0)
            else:
                tweets_target.append(1)
            all_tweets.append(tweet)
            
        return all_tweets, tweets_target