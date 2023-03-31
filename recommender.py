
import pickle
import requests
import pandas as pd
import streamlit as st


class Prepare_Data:

    def __init__(self, movie_file):
        
        self.recommended_movie_names = []

        self.recommended_movie_posters = []

        self.movies = pd.read_csv(movie_file)

        self.similarity = pickle.load(open('./similarity.pkl', 'rb'))
        

    def fetch_poster(self, movie_id):

        try:
            url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
                movie_id)
            data = requests.get(url)
            data = data.json()
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

            return full_path
        except:
            return "Errors"

    def recommend(self, movie, feature_name):

        try:
            index = self.movies[self.movies[feature_name] == movie].index[0]

            distances = sorted(
                list(enumerate(self.similarity[index])), reverse=True, key=lambda x: x[1])

            for score in distances[1:]:

                if score[1] > 0.25:

                    movie_id = self.movies.iloc[score[0]].movie_id
                    self.recommended_movie_posters.append(self.fetch_poster(movie_id))
                    self.recommended_movie_names.append(self.movies[feature_name].iloc[score[0]])

            return self.recommended_movie_names, self.recommended_movie_posters
        
        except:
            return "None"


class Display_result:

    def __init__(self, movie_file):

        st.header('Movie Recommender System')

        self.pr = Prepare_Data(movie_file)
        
        
    def prepare_data(self,feature_name):

        try:
            self.movie_list = self.pr.movies[feature_name].values
            self.selected_movie = st.selectbox(
            "Type or select a movie from the dropdown", self.movie_list)
        except:
            return "None"
        
    def display_posters(self,poster,val):

        columns = st.columns(val)
        for value in range(len(columns)):
            with columns[value]:
                st.image(poster[value])


    def run(self, feature_name):

        self.prepare_data(feature_name)

        try:
            if st.button('Show Recommendation'):
                recommended_movie_names, recommended_movie_posters = self.pr.recommend(
                    self.selected_movie, feature_name)
                
                if len(recommended_movie_posters) > 5:
                    self.display_posters(poster=recommended_movie_posters,val=5)
                elif len(recommended_movie_posters) <= 5:
                    self.display_posters(poster=recommended_movie_posters,val=len(recommended_movie_posters))      
        except:
          
            st.title('Error in Loading Files')


if __name__ == '__main__':

    dr = Display_result(r'S:\RNN-Python\movie_data.csv')
    dr.run(feature_name='title')
