import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


def setup_movie_clustering(movies_df):
    """Setup K-Means clustering for movie recommendations."""
    try:
        print("Setting up K-Means clustering...")

        # Encode categorical features
        le_genre = LabelEncoder()
        le_country = LabelEncoder()
        le_age = LabelEncoder()

        # Create feature matrix
        movies_df['genre_encoded'] = le_genre.fit_transform(movies_df['genre'].fillna('Unknown'))
        movies_df['country_encoded'] = le_country.fit_transform(movies_df['country'].fillna('Unknown'))
        movies_df['age_group_encoded'] = le_age.fit_transform(movies_df['age_group'].fillna('General'))

        feature_matrix = np.column_stack([
            movies_df['genre_encoded'].values,
            movies_df['country_encoded'].values,
            movies_df['age_group_encoded'].values,
            movies_df['released'].fillna(2000).values,
            movies_df['popular'].fillna(3.0).values,
            movies_df['runtime'].fillna(100).values
        ])

        # Scale and cluster
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        n_clusters = min(15, len(movies_df) // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        movies_df['cluster'] = kmeans.fit_predict(feature_matrix_scaled)

        print(f"Clustering completed with {n_clusters} clusters")
        return True

    except Exception as e:
        print(f"Clustering setup failed: {e}")
        movies_df['cluster'] = 0
        return False


def get_cluster_recommendations(movies_df, filtered_movies, limit=6):
    """Enhance recommendations using cluster information."""
    try:
        if 'cluster' not in movies_df.columns:
            return filtered_movies.head(limit)

        # Get cluster distribution from filtered movies
        cluster_counts = filtered_movies['cluster'].value_counts()

        # Get diverse recommendations from top clusters
        recommendations = []
        for cluster_id in cluster_counts.index[:3]:  # Top 3 clusters
            cluster_movies = filtered_movies[filtered_movies['cluster'] == cluster_id]
            recommendations.append(cluster_movies.head(2))

        # Combine and return
        if recommendations:
            result = pd.concat(recommendations).drop_duplicates(subset=['name']).head(limit)
            return result if not result.empty else filtered_movies.head(limit)
        else:
            return filtered_movies.head(limit)

    except Exception as e:
        print(f"Cluster recommendation failed: {e}")
        return filtered_movies.head(limit)