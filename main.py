import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
ratings = pd.read_csv("/Users/karthikvishal/Desktop/Projects/ml-100k/u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])

# Load movie titles from u.item
movies = pd.read_csv("/Users/karthikvishal/Desktop/Projects/ml-100k/u.item", sep="|", encoding='latin-1', header=None,
                     names=["movieId", "title", "release_date", "video_release", "IMDb_URL", "unknown", "Action",
                            "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                            "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

movies = movies[["movieId", "title"]]

# Merge data
data = pd.merge(ratings, movies, on="movieId")

# Create User-Movie matrix
pivot_table = data.pivot_table(index='userId', columns='title', values='rating')
pivot_table.fillna(0, inplace=True)

# Compute item-item similarity
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Recommendation function
def recommend(movie_name, num=5):
    if movie_name not in item_similarity_df:
        return "Movie not found in dataset"
    sim_scores = item_similarity_df[movie_name].sort_values(ascending=False)[1:num+1]
    return sim_scores

# Example output

x=st.text_input("Enter a movie name to get recommendations:")

st.write(recommend(x, 5))


