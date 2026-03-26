import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        return pd.merge(ratings, movies, on='movieId')
    except FileNotFoundError:
        print("Error: CSV files not found. Ensure they are in the same folder.")
        return None

def build_engine(df):
    # Popularity Filter (Min 50 ratings)
    movie_counts = df.groupby('title')['rating'].count()
    popular_movies = movie_counts[movie_counts >= 50].index
    df_filtered = df[df['title'].isin(popular_movies)]
    
    # User-Item Matrix
    pivot = df_filtered.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    # Cosine Similarity Matrix
    similarity = cosine_similarity(pivot)
    return pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

def get_recs(movie_name, similarity_df):
    matches = similarity_df.index[similarity_df.index.str.contains(movie_name, case=False)]
    if len(matches) == 0: return "No movie found."
    
    selected = matches[0]
    recs = similarity_df[selected].sort_values(ascending=False).iloc[1:6]
    return f"\nRecommendations for '{selected}':\n{recs}"

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        engine = build_engine(data)
        print("--- Movie Engine Ready ---")
        while True:
            query = input("\nEnter movie (or 'exit'): ")
            if query.lower() == 'exit': break
            print(get_recs(query, engine))