import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. Load the Data ---
# Note: Ensure these files are in the same folder as this script
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    print("Files loaded successfully!")
except FileNotFoundError:
    print("Error: movies.csv or ratings.csv not found in this folder.")
    exit()

# Merge datasets to get titles
df = pd.merge(ratings, movies, on='movieId')

# --- 2. Filter for Popularity ---
# We only keep movies with more than 50 ratings to avoid obscure "noise"
movie_counts = df.groupby('title')['rating'].count()
popular_movies = movie_counts[movie_counts >= 50].index
df_filtered = df[df['title'].isin(popular_movies)]

# --- 3. Create the User-Item Matrix ---
# Rows = Movie Titles, Columns = UserIDs
# This is Item-Based Collaborative Filtering
user_item_matrix = df_filtered.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# --- 4. Calculate Cosine Similarity ---
# This computes how similar every movie is to every other movie
item_similarity = cosine_similarity(user_item_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# --- 5. Recommendation Logic ---
def get_recommendations(movie_name):
    # Search for the movie title in our index (handles partial names)
    matches = item_similarity_df.index[item_similarity_df.index.str.contains(movie_name, case=False)]
    
    if len(matches) == 0:
        return None, "Movie not found. Try a more popular title!"
    
    selected_movie = matches[0] # Take the first close match
    
    # Get similarity scores and sort them
    similar_scores = item_similarity_df[selected_movie].sort_values(ascending=False)
    
    # Return top 5 (excluding the movie itself at index 0)
    return selected_movie, similar_scores.iloc[1:6]

# --- 6. Interactive Terminal Interface ---
print("\n" + "="*40)
print("  MOVIE RECOMMENDATION ENGINE (CF)  ")
print("="*40)

while True:
    user_query = input("\nEnter a movie name (or type 'quit' to exit): ")
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("Closing system. Goodbye!")
        break
        
    actual_title, results = get_recommendations(user_query)
    
    if actual_title is None:
        print(results)
    else:
        print(f"\nSince you liked: {actual_title}")
        print("-" * 30)
        print(results)
        print("-" * 30)