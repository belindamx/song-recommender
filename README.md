{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww17600\viewh12520\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Graph-Based Song Recommender\
\
A graph-based recommender that moves from familiar songs (Louvain Modularity, Degree Centrality) to new communities by following natural transition paths (Betweenness Centrality, Shortest Path) in the graph.\
\
Songs are represented as audio feature vectors and connected into a graph based on cosine similarity and k-nearest neighbors. Louvain modularity is used to detect communities; within communities, degree centrality identifies locally prominent songs. Betweenness centrality identifies bridge songs across communities. Recommendations begin with familiar songs from the source community, then transition to a nearby community via a bridge node using Dijkstra shortest path.\
\
## Dataset\
[Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)\
\
## Live App\
[Open Streamlit App](https://song-queue-generator-ftfayxrwrrtbjsy4d2gxu6.streamlit.app/)\
\
![Graph Visualization](graph.jpg)\
\
## How to Run\
1. Clone the repository\
2. Install dependencies with: `pip install -r requirements.txt`\
3. Start Neo4j locally (`bolt://localhost:7687`) and update username/password in the script\
4. Run the notebook or script to build the graph and generate recommendations}