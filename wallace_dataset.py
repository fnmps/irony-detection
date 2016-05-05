import csv
import re
import sqlite3

from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    def addRelationsFeatures(self, comment, comment_id):
        conn = sqlite3.connect('ironate-dk2.db')
        c = conn.cursor()
        result = c.execute("""SELECT redditor, subreddit FROM irony_comment WHERE id='%s' """ % comment_id)
        author, subreddit = result.fetchone()
        
        conn = sqlite3.connect('wallace_relations_small.db')
        c = conn.cursor()
        result = c.execute("""SELECT subject, verb, target FROM author_relations_small WHERE author='%s' AND subreddit='%s' """ % (author, subreddit))
        return list(result)
        
    #===============================================================================
    
    def __make_sql_list_str(self, ls):
        return "(" + ",".join([str(x_i) for x_i in ls]) + ")"
    
    def __grab_single_element(self, result_set, COL=0):
        return [x[COL] for x in result_set]
    
    def __get_labeled_thrice_comments(self, conn):
        c_id = conn.cursor()
        ''' get all ids for comments labeled >= 3 times '''
        c_id.execute(
            '''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;'''
        )
        thricely_labeled_comment_ids = self.__grab_single_element(c_id.fetchall())
        return thricely_labeled_comment_ids
    
    
    def __grab_comments(self, conn, comment_id_list):
        c_text = conn.cursor()
        comments_list = []
        for comment_id in comment_id_list:
            c_text.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
            segments = self.__grab_single_element(c_text.fetchall())
            comment = " ".join(segments)
            comments_list.append(comment.encode('utf-8').strip())
        return comments_list
    
    def __get_ironic_comment_ids(self, conn, labeler_id_str):
        cursor = conn.cursor()
        cursor.execute(
            '''select distinct comment_id from irony_label 
                where forced_decision=0 and label=1 and labeler_id in %s;''' % 
                labeler_id_str)
    
        ironic_comments = self.__grab_single_element(cursor.fetchall())
        return ironic_comments
    
    def __get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
    def comment_similarity(self, comment_id):
        conn = sqlite3.connect('ironate.db')
        c = conn.cursor()
        rows = c.execute("SELECT redditor, subreddit FROM irony_comment WHERE id=%s" % comment_id)
        row = rows.fetchone()
        author = row[0]
        subreddit = row[1]
        orig_comment = self.__grab_comments(conn, [comment_id])[0]
        conn = sqlite3.connect('wallace_dataset.db')
        c = conn.cursor()    
        comments = list(c.execute("SELECT distinct(comment_text) FROM irony_pastusercomment WHERE redditor='%s' AND subreddit='%s'" % (author, subreddit) ))
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform([orig_comment] + [c[0] for c in comments])
        similarities = (tfidf * tfidf.T).A
        result = []
        for simi in similarities:
            result.append(simi[0])
            
        return result
        
    
    def get_past_user_comments(self):        
        conn = sqlite3.connect('wallace_dataset.db')
        comment_id_list = self.__get_labeled_thrice_comments(conn)
        c = conn.cursor()
        authors = set()
        for comment_id in comment_id_list:
            c.execute("SELECT DISTINCT redditor FROM irony_comment WHERE id=%s" % comment_id)
            authors.add(c.fetchone()[0])
        
        c= conn.cursor()
                
        rows = c.execute("select redditor, comment_text, subreddit from irony_pastusercomment")
        data = dict()
        for author, comment, subreddit in rows:
            if author in authors:
                if author in data:
                    data[author].append( (subreddit, comment) )
                else:
                    data[author] = [(subreddit, comment)]
        return data
    
    def __get_relations(self):        
        conn = sqlite3.connect('wallace_relations_small.db')
        c = conn.cursor()
        author_relations = dict()
        authors = c.execute("SELECT DISTINCT author FROM author_relations_small")

        for author in authors:
            result = c.execute("SELECT subreddit, subject, verb, target FROM author_relations_small WHERE author='%s'" % author)
            author_relations[author] = result
        
        return author_relations
    
    
    def get_data(self, affective_norms=False, emoticons=False, punctuation=False, relations=False):
        conn = sqlite3.connect('ironate.db')
        
        labelers_of_interest = [2,4,5,6]
        labeler_id_str = self.__make_sql_list_str(labelers_of_interest)
        
        comment_id_list = self.__get_labeled_thrice_comments(conn)
        ironic_comments_ids = self.__get_ironic_comment_ids(conn, labeler_id_str)
        if affective_norms:
            for row in csv.DictReader(open("Ratings_Warriner_et_al.csv")): 
                self.affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
        
        ironic_comments = []
        not_ironic_comments = []
        comments_target = []
        all_comments = []
        
        for comment_id in comment_id_list:
            comment = self.__grab_comments(conn, [comment_id])[0]
            if affective_norms:
                comment = self.__addAffectiveNormsFeature(comment)
            if punctuation:
                comment = self.__addPunctFeatures(comment)
            if emoticons:
                comment = self.__addEmoticonsFeatures(comment)
            if relations:
                comment = self.__addRelationsFeatures(comment, comment_id)
            
            if comment_id in ironic_comments_ids:
                ironic_comments.append(comment)
                comments_target.append(0)
            else:
                not_ironic_comments.append(comment)
                comments_target.append(1)
            all_comments.append(comment)
            
        return all_comments, comments_target