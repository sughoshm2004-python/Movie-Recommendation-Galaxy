import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. Load Data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
df = pd.merge(ratings, movies, on='movieId')

# 2. Filter for Popularity (We only map movies people actually know)
movie_counts = df.groupby('title')['rating'].count()
popular_movies = movie_counts[movie_counts >= 50].index
df_filtered = df[df['title'].isin(popular_movies)]

# 3. Create the Matrix
user_item_matrix = df_filtered.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# 4. Dimensionality Reduction (t-SNE)
# This "squashes" 610 users down to 2 coordinates (x, y)
print("Analyzing the movie galaxy... please wait.")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(user_item_matrix)

# 5. Prepare Plotting Data
tsne_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
tsne_df['title'] = user_item_matrix.index

# Join with movies to get the primary Genre for coloring
tsne_df = tsne_df.merge(movies[['title', 'genres']], on='title')
tsne_df['Main_Genre'] = tsne_df['genres'].apply(lambda x: x.split('|')[0]) # Get the first genre listed

# 6. Visualize
plt.figure(figsize=(14, 10))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='Main_Genre', palette='viridis', alpha=0.7, s=100)

# Label a few famous movies to see where they land
labels = ['Toy Story (1995)', 'Matrix, The (1999)', 'Pulp Fiction (1994)', 'Forrest Gump (1994)', 'Lion King, The (1994)']
for i, row in tsne_df.iterrows():
    if row['title'] in labels:
        plt.annotate(row['title'], (row['x'], row['y']), fontsize=10, weight='bold')

plt.title("Data Analysis: How Movies Cluster Based on User Ratings", fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()