"""
Run once to generate data/nodes.parquet and data/edges.parquet.

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import community as community_louvain

# ── Load & Clean ──────────────────────────────────────────────────────────────

spotify_df = pd.read_csv('dataset.csv', index_col=0)
spotify_df = spotify_df.dropna(subset=['album_name'])

# remove track_id duplicates (same song across different album versions) keeping most popular one
spotify_df.sort_values(by='popularity', ascending=False, inplace=True)
spotify_df.drop_duplicates(subset='track_id', inplace=True)
spotify_df.drop_duplicates(subset=['track_name', 'artists'], inplace=True)
spotify_df.reset_index(drop=True, inplace=True)

print(f'Songs after dedup: {len(spotify_df)}')

# ── Feature Engineering ───────────────────────────────────────────────────────

# circular encoding for key (0-11) to preserve circular musical relationship
# e.g. key 11 (B) should be close to key 0 (C), not far away
spotify_df['key_sin'] = np.sin(2 * np.pi * spotify_df['key'] / 12)
spotify_df['key_cos'] = np.cos(2 * np.pi * spotify_df['key'] / 12)

song_features = spotify_df[[
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'mode', 'key_sin', 'key_cos'
]]

scaler = StandardScaler()
spotify_scaled = scaler.fit_transform(song_features)

# apply feature weights after scaling
# energy and tempo weighted up: most perceptually jarring when mismatched
# valence weighted up: mood is strongly felt by listeners
# liveness, key, mode weighted down: less perceptually salient
feature_weights = {
    'danceability':     1.0,
    'energy':           1.5,
    'loudness':         1.0,
    'speechiness':      1.0,
    'acousticness':     1.0,
    'instrumentalness': 1.0,
    'liveness':         0.5,
    'valence':          1.2,
    'tempo':            1.5,
    'mode':             0.8,
    'key_sin':          0.5,
    'key_cos':          0.5,
}
weights = np.array([feature_weights[f] for f in song_features.columns])
spotify_scaled_weighted = spotify_scaled * weights

# L2 normalize weighted features so euclidean distance == cosine distance
# ball_tree with euclidean on normalized vectors is much faster than brute-force cosine
spotify_normalized = normalize(spotify_scaled_weighted)

# ── KNN Graph ─────────────────────────────────────────────────────────────────

print('Running KNN...')

knn = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='ball_tree')
knn.fit(spotify_normalized)
distances, indices = knn.kneighbors(spotify_normalized)

# euclidean on normalized vectors back to cosine distance: cos_dist = euclidean^2 / 2
cos_distances = distances ** 2 / 2

edges = []
for i in range(len(spotify_df)):
    src_id = spotify_df.iloc[i]['track_id']
    for j in range(1, 6):
        nbr_idx = indices[i][j]
        nbr_id  = spotify_df.iloc[nbr_idx]['track_id']
        cos_dist = float(cos_distances[i][j])
        edges.append({
            'source':     src_id,
            'target':     nbr_id,
            'similarity': float(1 - cos_dist),
            'cost':       cos_dist
        })

edges_df = pd.DataFrame(edges)
print(f'Edges: {len(edges_df)}')

# ── Build NetworkX Graph ──────────────────────────────────────────────────────

print('Building graph...')

G = nx.Graph()
for _, row in spotify_df.iterrows():
    G.add_node(row['track_id'])
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], cost=row['cost'], similarity=row['similarity'])

# ── Louvain Community Detection ───────────────────────────────────────────────

print('Running Louvain...')

partition = community_louvain.best_partition(G, weight='similarity')
print(f'Communities: {len(set(partition.values()))}')

# ── Degree Centrality ─────────────────────────────────────────────────────────

print('Running degree centrality...')

degree = nx.degree_centrality(G)

# ── Approximate Betweenness Centrality ───────────────────────────────────────

# k=100 matches the samplingSize used in the Neo4j version
print('Running betweenness (k=100 sample, ~2-3 min)...')

betweenness = nx.betweenness_centrality(G, k=100, seed=0)

# ── Export Parquet ────────────────────────────────────────────────────────────

os.makedirs('data', exist_ok=True)

nodes = spotify_df[[
    'track_id', 'track_name', 'artists', 'track_genre', 'popularity',
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'mode'
]].copy().rename(columns={'track_genre': 'genre'})

nodes['community']   = nodes['track_id'].map(partition)
nodes['degree']      = nodes['track_id'].map(degree)
nodes['betweenness'] = nodes['track_id'].map(betweenness)

nodes.to_parquet('data/nodes.parquet', index=False)
print(f'Saved data/nodes.parquet ({len(nodes)} nodes)')

edges_df.to_parquet('data/edges.parquet', index=False)
print(f'Saved data/edges.parquet ({len(edges_df)} edges)')

# ── Verify ────────────────────────────────────────────────────────────────────

n = pd.read_parquet('data/nodes.parquet')
e = pd.read_parquet('data/edges.parquet')
print(f'Nodes: {len(n)}, Edges: {len(e)}')
print(f'Communities: {n["community"].nunique()}')
print(f'Songs with degree:      {n["degree"].notna().sum()}')
print(f'Songs with betweenness: {n["betweenness"].notna().sum()}')
