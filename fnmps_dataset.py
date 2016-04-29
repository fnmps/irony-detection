import csv
import re
import sqlite3
import numpy as np

class dataset():
    
    def __init__(self):
        self.affective = dict( )
        
    #===============================================================================
    def __addPunctFeatures(self, comment):
    
        #FEATURES
        question_mark_RE_str = '\?'
        exclamation_point_RE_str = '\!'
        interrobang_RE_str = '[\?\!]{2,}'
         
        if len(re.findall(r'%s' % exclamation_point_RE_str, comment)) > 0:
            comment += " PUNCxEXCLAMATION_POINT"
        if len(re.findall(r'%s' % question_mark_RE_str, comment)) > 0:
            comment += " PUNCxQUESTION_MARK"
        if len(re.findall(r'%s' % interrobang_RE_str, comment)) > 0:
            comment += " PUNCxINTERROBANG"
        if any([len(s) > 2 and str.isupper(s) for s in comment.split(" ")]):
            comment += " PUNCxUPPERCASE"
        return comment

    def __addEmoticonsFeatures(self, comment):   
        emoticon_RE_str = '(?::|;|=|X|x)(?:-)?(?:\)|\(|D|P)'
         
        if len(re.findall(r'%s' % emoticon_RE_str, comment)) > 0:
            comment += " PUNCxEMOTICON"
        return comment
    
    
    
    def __addAffectiveNormsFeature(self, comment):   
        valence = 0
        arousal = 0
        dominance = 0
        counter = 0
        for word in comment.split(' '):
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
                comment += " LOW_VALENCE"
            elif valence < 7:
                comment += " MEDIUM_VALENCE"
            else:
                comment += " HIGH_VALENCE"
            if arousal < 3:
                comment += " LOW_AROUSAL"
            elif arousal < 7:
                comment += " MEDIUM_AROUSAL"
            else:
                comment += " HIGH_AROUSAL"
            if dominance < 3:
                comment += " LOW_DOMINANCE"
            elif dominance < 7:
                comment += " MEDIUM_DOMINANCE"
            else:
                comment += " HIGH_DOMINANCE"
        return comment
    
    #===============================================================================
    
    def __get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
    def get_data(self, author=False, subreddit=False, affective_norms=False, emoticons=False, punctuation=False):
        conn = sqlite3.connect('fnmps_dataset.db')
        conn.text_factory = str
        cursor = conn.cursor()
        cursorSubreddit = conn.cursor()
        
        
        if affective_norms:
            for row in csv.DictReader(open("Ratings_Warriner_et_al.csv")): 
                self.affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
        
        comments_subreddit = dict()
        if subreddit:
            results = cursorSubreddit.execute("SELECT id, subreddit from author_comments")
            for comment_id, subreddit in results:
                comments_subreddit[comment_id] = subreddit
        
        rows = cursor.execute("SELECT comment_id, author, comment_segment, label from labeled_comments")
        comments_target = []
        all_comments = []
        
        for comment_id, author, comment_segment, label in rows:
            comment = comment_segment
            if affective_norms:
                comment = self.__addAffectiveNormsFeature(comment)
            if punctuation:
                comment = self.__addPunctFeatures(comment)
            if emoticons:
                comment = self.__addEmoticonsFeatures(comment)
            if subreddit:
                subreddit = comments_subreddit[comment_id]
                comment += " SUBREDDIT"+subreddit
            if author:
                comment += " AUTHOR"+author
            
            if label == 'IRONIC':
                comments_target.append(0)
            else:
                comments_target.append(1)
            all_comments.append(comment)
            
        return all_comments, comments_target