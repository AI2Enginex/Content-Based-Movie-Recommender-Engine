

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from funcs import Pre_processing_functions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Word_Vectors:

    def __init__(self,file_name):
        
        self.new_df = pd.read_csv(file_name)
        self.cv = CountVectorizer(max_features=5000, stop_words='english')
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

    def create_vector(self, feature_name):

        return cosine_similarity(self.cv.fit_transform(self.new_df[feature_name]).toarray())
       
    def tf_idf_vector(self, feature_name):

        return cosine_similarity(self.cv.fit_transform(self.new_df[feature_name]).toarray())
        
    def create_word2vec(self, word_arr, most_similar, window_size, val):

        return Word2Vec(word_arr, min_count=most_similar,  window=window_size, sg=val)
        

class Recommend_Movies(Word_Vectors):

    def __init__(self, file_name):

        super().__init__(file_name)

        self.file = self.new_df
        

    def recommend(self, movie_name, feature1, feature2, file_name):

        try:
            similarity = self.tf_idf_vector(feature2)
            index = self.file[self.file[feature1] == movie_name].index[0]
            distances = sorted(
                list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
            for i in distances[1:]:

                if i[1] > 0.25:
                    print(self.file[feature1].iloc[i[0]])

            pickle.dump(similarity, open(file_name, 'wb'))
        except:
            print('invalid parameters passed.....')

class Compute_Scores(Word_Vectors):

    def __init__(self, file):

        super().__init__(file)
        
        self.result = self.new_df
        self.fn = Pre_processing_functions()
        self.my_dict = dict()
        
    def generate_vectors(self, most_similar, window_size, sg,feature_name):

        word_arr = self.fn.process_text(df=self.result, feature=feature_name)
        return self.create_word2vec(word_arr, most_similar, window_size, sg)
        
    def compute_similarity(self,most_similar, window_size,sg,feature_name,value1,value2):
        
        try:
            self.gensim_model = self.generate_vectors(most_similar, window_size, sg,feature_name)
            return self.gensim_model.wv.similarity(value1.lower(),value2.lower())
        except Exception:
            return f'worng parameters passed'
        
    def find_similar(self,val_to_find,most_similar, window_size, sg,feature_name):
            
        try:
            self.gensim_model = self.generate_vectors(most_similar, window_size, sg,feature_name)
            return self.fn.create_df(obj_val=dict(self.gensim_model.wv.most_similar(val_to_find.lower())))
        except Exception as e:
            return f'worng parameters passed'
        

if __name__ == '__main__':
   
  pass

   
