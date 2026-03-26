Movie Galaxy: Collaborative Filtering & Latent Space Analysis 🎬🌌
A Data Science project that applies Unsupervised Learning and Dimensionality Reduction to map human movie preferences based on 100,000+ user ratings.

🚀 Overview
This project explores how high-dimensional user behavior can be condensed into a 2D "Movie Galaxy." By using Collaborative Filtering, the system predicts user interests without needing any information about the movie plots or actors—relying purely on mathematical similarity.

🛠️ Tech Stack
Language: Python 3.x

Environment: VS Code / Anaconda

Libraries: pandas, scikit-learn, matplotlib, seaborn

Algorithms: Item-Item Collaborative Filtering, Cosine Similarity, t-SNE

📊 Analytical Highlights
1. The Recommendation Engine
The system uses Cosine Similarity to find the "distance" between movie vectors.

Handling Sparsity: Managed a dataset where ~98% of potential ratings were missing.

Popularity Filtering: Only movies with 50+ ratings were analyzed to ensure statistical significance.

2. t-SNE Galaxy Mapping
Using t-SNE (t-Distributed Stochastic Neighbor Embedding), I projected the 610-dimensional user rating space into a 2D map.

Finding: Even without "knowing" the genre, the math naturally clustered similar movies (e.g., Disney animations or 90s Action blockbusters) together in specific "neighborhoods."

3. Genre Centroid Analysis
I calculated the Mathematical Centroid (center of mass) for each genre to quantify the "distance" between different audience tastes.


Project structure
├── recommender_engine.py      # Interactive recommendation script
├── movie_galaxy_analysis.py   # t-SNE mapping and centroid logic
├── movie_galaxy_plot.png      # Final visualization of the clusters
├── requirements.txt           # Required Python libraries
└── .gitignore                 # Filters out large CSV datasets
