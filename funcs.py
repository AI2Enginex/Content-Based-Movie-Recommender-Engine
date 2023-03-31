
import re
import ast
import nltk

import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
"""
This file contains all the preprocessing functions 
required to prepare the data , these functions are 
used in prepare_data.ipynb file

"""


class Pre_processing_functions:

    def __init__(self):

        self.vocab_lst = list()
        
    def read_files(self,file_name):
        try:
            df = pd.read_csv(file_name)
            return df
        except:
            pass

    def merge_data(self, df1,df2,on_feature):

        try:
            return df1.merge(df2, on=on_feature)
        except:
            return "None"

    def select_features(self, df, *args):

        try:
            return df[[i for i in args]]
        except:
            return "None"

    def convert(self,text,key):

        try:
            L1 = []
            for i in ast.literal_eval(text):
                L1.append(i[key])
            return L1
        except:
            return "None"

    def fetch_(self,text,key,filed_name,value):

        try:
            L2 = []
            for i in ast.literal_eval(text):
                if i[filed_name] == value:
                    L2.append(i[key])
            return L2
        except:
            return "None"

    def drop_many(self, data, *args):

        try:
            data = data.drop(columns=[i for i in args])
            return data
        except:
            return "None"

    def collapse(self, text, str1, str2):

        try:
            L3 = []
            for i in text:
                L3.append(i.replace(str1, str2))
            return L3
        except:
            return "None"
        
    def count_(self,feature):
        
        try:
            total = Counter(feature)
            return total.items()
        except:

            return "error"
        
    def process_text(self,df,feature):

        try:
            for message in df[feature]:

                word = re.sub('[^a-zA-Z]',' ' , message)
                review = word.split()
                stop_words = [word for word in review if word not in 
                              stopwords.words('english')]
                self.vocab_lst.append(stop_words)
            return self.vocab_lst
        except:
            return "Error"
        
    def create_df(self,obj_val):

        try:
            return  pd.DataFrame.from_dict(obj_val , orient='index',columns=['score'])
        except:
            return "Error"

if __name__ == '__main__':

    pass
