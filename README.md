# Graph-Based Song Recommender

A graph-based recommender that moves from familiar songs (Louvain Modularity, Degree Centrality) to new communities by following natural transition paths (Betweenness Centrality, Shortest Path) in the graph.

Songs are loaded as graph nodes, with edge connections based on cosine similarity of audio feature vectors, for 5 nearest neighbors. Louvain modularity is used to detect communities and within communities, degree centrality identifies locally representative songs. Betweenness centrality identifies bridge songs across communities. Recommendations begin with familiar songs from the source community, then transition to a nearby community via a bridge node using Dijkstra shortest path.

Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

Live App: [View App](https://song-path.streamlit.app)

Recommendation Flow:
![Flow Image](assets/flow_img.png)

## How to Run
1. Clone repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Build the graph: `python build_graph.py`  
4. Launch the app: `streamlit run app.py`

---

Note: The `neo4j_version` notebook contains the original architecture using Neo4j and Docker. The Streamlit app was built for demonstrative purposes using NetworkX. 