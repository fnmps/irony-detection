import csv
import re
import sqlite3
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
    
    
    def __get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
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