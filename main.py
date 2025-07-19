import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load ratings
ratings = pd.read_csv(
    "./u.data",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

# Load movie titles from u.item
movies = pd.read_csv(
    "./u.item",
    sep="|",
    encoding='latin-1',
    header=None,
    names=["movieId", "title", "release_date", "video_release", "IMDb_URL", "unknown", "Action",
           "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
           "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
)

movies = movies[["movieId", "title"]]

# Merge ratings and movies
data = pd.merge(ratings, movies, on="movieId")

# Create User-Movie matrix
pivot_table = data.pivot_table(index='userId', columns='title', values='rating')
pivot_table.fillna(0, inplace=True)

# Compute item-item cosine similarity
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Fuzzy matching function
def find_closest_match(movie_input):
    movie_titles = item_similarity_df.columns
    match, score = process.extractOne(movie_input, movie_titles)
    return match if score >= 60 else None

# Recommendation function
def recommend(movie_name, num=5):
    if movie_name not in item_similarity_df:
        return "Movie not found in dataset."
    sim_scores = item_similarity_df[movie_name].sort_values(ascending=False)[1:num+1]
    return sim_scores

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.markdown("Type a movie name to get similar movie recommendations.")

user_input = st.text_input("Enter a movie name:")

if user_input:
    matched_movie = find_closest_match(user_input)
    if matched_movie:
        st.success(f"Showing results for: **{matched_movie}**")
        results = recommend(matched_movie, 5)
        st.write("Top 5 similar movies:")
        st.table(results)
    else:
        st.error("❌ Sorry, no close match found in the dataset.")
