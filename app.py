import html as html_lib
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import re

st.set_page_config(
    page_title="Song Path Generator",
    page_icon="🎵",
    layout="centered"
)

# ── Language Detection ────────────────────────────────────────────────────────
# keeps queues linguistically coherent (no random spanish songs in an english queue)

LATIN_GENRES = {
    'spanish', 'latin', 'latino', 'reggaeton', 'salsa', 'tango',
    'sertanejo', 'forro', 'pagode', 'samba', 'mpb', 'brazil',
}

OTHER_NON_ENGLISH_GENRES = {
    'cantopop', 'mandopop', 'j-pop', 'j-rock', 'j-idol', 'j-dance',
    'k-pop', 'anime', 'turkish', 'french', 'german', 'swedish',
    'malay', 'iranian', 'indian', 'romance', 'world-music', 'afrobeat',
}

_LATIN_ARTISTS_RE = re.compile(
    r'\b(bad bunny|karol g|j balvin|daddy yankee|ozuna|maluma|anuel aa|'
    r'jhayco|feid|sech|myke towers|rauw alejandro|farruko|nicky jam|'
    r'zion|chencho corleone|manuel turizo|camilo|paulo londra|cnco|'
    r'becky g|natti natasha|lunay|mora|dei v|arcangel|de la ghetto|'
    r'wisin|yandel|cosculluela|alex rose|lyanno|brray|piso 21|'
    r'chris jedi|cris mj|quevedo|bizarrap|sebastián yatra|'
    r'rosalía|c tangana|bad gyal)\b',
    re.IGNORECASE
)

_SPANISH_WORDS_RE = re.compile(
    r'\b(el|la|los|las|un|una|de|del|que|por|con|para|como|todo|pero|'
    r'hay|fue|ser|muy|sin|sobre|entre|cuando|donde|aunque|porque|desde|'
    r'hasta|tambien|solo|siempre|nunca|algo|nada|cada|otro|otra|mismo|'
    r'misma|hace|querer|saber|llegar|pasar|seguir|llamar|venir|pensar|'
    r'poner|parecer|quedar|creer|hablar|llevar|dejar|sentir|conocer|'
    r'vivir|decir|ver|dar|estar|agua|amor|vida|tiempo|mundo|casa|forma|'
    r'parte|lugar|dia|vez|noche|ciudad|pueblo|camino|tierra|cielo|'
    r'corazon|mente|gente|hijo|madre|padre|hermano|amigo|te|tu|mi|'
    r'yo|no|si|me|lo|le|se|al|es|en|su|ya|ni|o)\b',
    re.IGNORECASE
)

def _has_spanish_words(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    if not words:
        return False
    matches = sum(1 for w in words if _SPANISH_WORDS_RE.match(w))
    return matches / len(words) > 0.35

def detect_language(track_name, genre, artists=''):
    if genre in LATIN_GENRES:
        return 'latin'
    if _LATIN_ARTISTS_RE.search(str(artists)):
        return 'latin'
    if _has_spanish_words(str(track_name)):
        return 'latin'
    if genre in OTHER_NON_ENGLISH_GENRES:
        return 'other'
    if not str(track_name).isascii() or not str(artists).isascii():
        return 'other'
    return 'english'

def allowed_in_queue(track_name, genre, artists, input_language):
    song_lang = detect_language(track_name, genre, artists)
    if input_language == 'english':
        return song_lang == 'english'
    elif input_language == 'latin':
        return song_lang in ('english', 'latin')
    else:
        return song_lang in ('english', 'other')

# ── Genre Compatibility ───────────────────────────────────────────────────────
# prevents jarring community hops (pop → death metal, pop → children, etc.)

def _genre_family(genre):
    g = str(genre).lower()
    if 'children' in g or 'kids' in g:
        return 'children'
    if 'death' in g or 'black' in g or 'doom' in g or 'grind' in g:
        return 'extreme-metal'
    if 'metal' in g or 'core' in g:
        return 'metal'
    if 'classical' in g or 'opera' in g or 'orchestr' in g:
        return 'classical'
    if 'jazz' in g or 'blues' in g or 'bossa' in g:
        return 'jazz'
    if 'country' in g or 'folk' in g or 'americana' in g or 'bluegrass' in g:
        return 'country'
    if 'hip' in g or 'rap' in g or 'trap' in g or 'r-n-b' in g or 'soul' in g or 'funk' in g:
        return 'hip-hop'
    if 'latin' in g or 'reggaeton' in g or 'salsa' in g or 'tango' in g or 'sertanejo' in g:
        return 'latin'
    if 'k-pop' in g or 'j-pop' in g or 'anime' in g or 'cantopop' in g or 'mandopop' in g:
        return 'asian-pop'
    if 'rock' in g or 'punk' in g or 'grunge' in g or 'indie' in g or 'alternative' in g or 'emo' in g or 'hardcore' in g or 'post-' in g or 'shoegaze' in g:
        return 'rock'
    if 'pop' in g or 'dance' in g or 'electro' in g or 'edm' in g or 'techno' in g or 'house' in g or 'synth' in g:
        return 'pop'
    return 'other'

_FAMILY_COMPAT = {
    'pop':          {'pop', 'rock', 'hip-hop', 'latin', 'asian-pop'},
    'rock':         {'rock', 'pop', 'metal', 'hip-hop', 'country'},
    'metal':        {'metal', 'rock', 'extreme-metal'},
    'extreme-metal':{'extreme-metal', 'metal'},
    'hip-hop':      {'hip-hop', 'pop', 'rock', 'latin'},
    'country':      {'country', 'rock', 'jazz'},
    'jazz':         {'jazz', 'country', 'classical'},
    'classical':    {'classical', 'jazz'},
    'latin':        {'latin', 'pop', 'hip-hop'},
    'asian-pop':    {'asian-pop', 'pop'},
    'children':     {'children'},
    'other':        {'other', 'pop', 'rock'},
}

def genre_compatible(input_genre, target_genre):
    return _genre_family(target_genre) in _FAMILY_COMPAT.get(_genre_family(input_genre), {'other'})

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    nodes = pd.read_parquet('data/nodes.parquet')
    edges = pd.read_parquet('data/edges.parquet')

    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(row['track_id'], **row.to_dict())
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], cost=row['cost'], similarity=row['similarity'])

    # precompute neighbor community sets for fast bridge lookups
    neighbor_communities = {
        node: {G.nodes[n].get('community') for n in G.neighbors(node)}
        for node in G.nodes
    }

    return G, nodes, neighbor_communities

@st.cache_data
def load_community_centroids():
    nodes = pd.read_parquet('data/nodes.parquet')
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'mode'
    ]
    return {
        community: {col: group[col].mean() for col in feature_cols}
        for community, group in nodes.groupby('community')
    }

@st.cache_data
def load_community_labels():
    # label each community by its most common genre
    nodes = pd.read_parquet('data/nodes.parquet')
    return (
        nodes.groupby('community')['genre']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

# ── Queue Generation ──────────────────────────────────────────────────────────

def find_target_community(input_community, community_centroids, G, nodes,
                          neighbor_communities, adventurousness=2):
    input_centroid = np.array(list(community_centroids[input_community].values()))

    distances = [
        (community, np.linalg.norm(input_centroid - np.array(list(c.values()))))
        for community, c in community_centroids.items()
        if community != input_community
    ]
    distances.sort(key=lambda x: x[1])
    ranked_communities = [c for c, _ in distances]

    # find communities reachable via high-betweenness bridge nodes
    input_songs = set(nodes[nodes['community'] == input_community]['track_id'])
    valid_communities = set()
    for node_id in input_songs:
        for neighbor_id in G.neighbors(node_id):
            if (G.nodes[neighbor_id].get('community') == input_community
                    and G.nodes[neighbor_id].get('betweenness') is not None):
                valid_communities |= neighbor_communities[neighbor_id] - {input_community}

    input_genre = community_labels.get(input_community, '')
    valid_ranked = [
        c for c in ranked_communities
        if c in valid_communities
        and genre_compatible(input_genre, community_labels.get(c, ''))
    ]
    if not valid_ranked:
        # genre filter too strict — fall back to any reachable community
        valid_ranked = [c for c in ranked_communities if c in valid_communities]
    if not valid_ranked:
        raise ValueError(f'No reachable target community from community {input_community}')

    # adventurousness controls where in the ranked list we sample from
    # 1 = closest communities, 5 = furthest communities
    n = len(valid_ranked)
    rng = np.random.default_rng()
    if adventurousness == 1:
        pool = valid_ranked[:max(1, n // 5)]
    elif adventurousness == 2:
        pool = valid_ranked[:max(1, n // 3)]
    elif adventurousness == 3:
        pool = valid_ranked[n // 4: 3 * n // 4]
    elif adventurousness == 4:
        pool = valid_ranked[n // 2:]
    else:
        pool = valid_ranked[max(0, n - n // 5):]

    return int(pool[rng.integers(0, len(pool))])


def find_bridge_song(input_community, target_community, G, neighbor_communities):
    candidates = [
        (node_id, data['betweenness'], data.get('community'))
        for node_id, data in G.nodes(data=True)
        if data.get('community') in (input_community, target_community)
        and data.get('betweenness') is not None
        and input_community in neighbor_communities[node_id]
        and target_community in neighbor_communities[node_id]
    ]

    if not candidates:
        raise ValueError(f'No bridge song found between communities {input_community} and {target_community}')

    candidates.sort(key=lambda x: x[1], reverse=True)
    rng = np.random.default_rng()
    chosen = candidates[:3][rng.integers(0, min(3, len(candidates)))]
    return chosen[0], int(chosen[2])


def generate_queue(input_track_id, G, nodes, neighbor_communities, community_centroids,
                   queue_length=8, max_artist_appearances=2, adventurousness=2, min_popularity=35):
    if input_track_id not in G.nodes:
        raise ValueError(f'No song found with track_id: {input_track_id}')

    input_data      = G.nodes[input_track_id]
    input_community = int(input_data['community'])
    input_language  = detect_language(
        input_data.get('track_name', ''),
        input_data.get('genre', ''),
        input_data.get('artists', '')
    )

    def ok(track_name, genre, artists):
        return allowed_in_queue(track_name, genre, artists, input_language)

    target_community            = find_target_community(input_community, community_centroids,
                                                        G, nodes, neighbor_communities, adventurousness)
    bridge_id, bridge_community = find_bridge_song(input_community, target_community, G, neighbor_communities)

    rng = np.random.default_rng()

    # split queue into three explicit phases so roles are unambiguous
    n_start = queue_length // 2           # even split, start gets the extra if odd remainder
    n_dest  = queue_length - 1 - n_start  # 1 slot reserved for bridge

    def bfs_collect(seed_ids, community_id, n, used_ids):
        # BFS through neighbors sorted by similarity + degree, staying in community_id
        result, visited = [], set(seed_ids) | set(used_ids)
        # dedup by track_name — same song can appear under multiple track_ids
        used_names = {G.nodes[tid].get('track_name') for tid in visited if tid in G.nodes}
        queue_bfs = list(seed_ids)
        while len(result) < n and queue_bfs:
            curr = queue_bfs.pop(0)
            for nbr in sorted(G.neighbors(curr),
                              key=lambda nb: -(G[curr][nb].get('similarity', 0) * 0.7
                                               + G.nodes[nb].get('degree', 0) * 0.3)):
                if nbr in visited:
                    continue
                visited.add(nbr)
                nd = G.nodes[nbr]
                if nd.get('community') != community_id:
                    continue
                if nd.get('track_name') in used_names:
                    continue
                if (nd.get('popularity') or 0) < min_popularity:
                    continue
                if not ok(nd.get('track_name', ''), nd.get('genre', ''), nd.get('artists', '')):
                    continue
                result.append(nbr)
                used_names.add(nd.get('track_name'))
                queue_bfs.append(nbr)
                if len(result) >= n:
                    break
        return result

    used = {bridge_id}

    # phase 1: seed + similar songs in input community
    start_ids = [input_track_id] + bfs_collect([input_track_id], input_community, n_start - 1, used | {input_track_id})
    used |= set(start_ids)

    # phase 2: bridge song (already found above)

    # phase 3: songs in target community reachable from bridge
    dest_ids = bfs_collect([bridge_id], target_community, n_dest, used)

    def make_row(tid, role):
        d = G.nodes[tid]
        return {
            'track_id':   tid,
            'track_name': d.get('track_name'),
            'artist':     d.get('artists'),
            'genre':      d.get('genre'),
            'community':  d.get('community'),
            'energy':     d.get('energy'),
            'valence':    d.get('valence'),
            'tempo':      d.get('tempo'),
            'role':       role,
        }

    rows = [make_row(tid, 'start') for tid in start_ids]
    rows.append(make_row(bridge_id, 'bridge'))
    rows += [make_row(tid, 'destination') for tid in dest_ids]

    final_path = pd.DataFrame(rows).reset_index(drop=True)

    # pad to queue_length if bfs_collect came up short (filters too strict)
    if len(final_path) < queue_length:
        used_ids    = set(final_path['track_id'])
        used_names  = set(final_path['track_name'].dropna())
        pad_queue   = list(final_path['track_id'])
        pad_visited = set(pad_queue)
        while len(final_path) < queue_length and pad_queue:
            curr = pad_queue.pop(0)
            for nbr in sorted(G.neighbors(curr), key=lambda nb: -G[curr][nb].get('similarity', 0)):
                if nbr in pad_visited or nbr in used_ids:
                    continue
                pad_visited.add(nbr)
                nd = G.nodes[nbr]
                if nd.get('track_name') in used_names:
                    continue
                if (nd.get('popularity') or 0) < min_popularity:
                    continue
                if not ok(nd.get('track_name', ''), nd.get('genre', ''), nd.get('artists', '')):
                    continue
                new_row = make_row(nbr, 'destination')
                final_path = pd.concat([final_path, pd.DataFrame([new_row])], ignore_index=True)
                used_ids.add(nbr)
                used_names.add(nd.get('track_name'))
                pad_queue.append(nbr)
                break

    # artist repeat limiting — preserve role when replacing
    artist_counts  = final_path['artist'].value_counts()
    excess_artists = artist_counts[artist_counts > max_artist_appearances].index.tolist()
    for artist in excess_artists:
        to_replace = final_path[final_path['artist'] == artist].index[max_artist_appearances:]
        for idx in to_replace:
            curr_id   = final_path.loc[idx, 'track_id']
            curr_comm = final_path.loc[idx, 'community']
            used_ids  = set(final_path['track_id'])
            replacements = [
                n for n in G.neighbors(curr_id)
                if G.nodes[n].get('artists') != artist
                and n not in used_ids
                and G.nodes[n].get('community') == curr_comm
                and ok(G.nodes[n].get('track_name', ''), G.nodes[n].get('genre', ''), G.nodes[n].get('artists', ''))
            ]
            if replacements:
                r_id = replacements[rng.integers(0, len(replacements))]
                d    = G.nodes[r_id]
                final_path.loc[idx, ['track_id', 'track_name', 'artist', 'genre', 'community',
                                     'energy', 'valence', 'tempo']] = [
                    r_id, d.get('track_name'), d.get('artists'), d.get('genre'),
                    d.get('community'), d.get('energy'), d.get('valence'), d.get('tempo')
                ]

    journey = {
        'input_community':  input_community,
        'bridge_community': bridge_community,
        'target_community': target_community,
    }

    return final_path, journey

# ── Startup ───────────────────────────────────────────────────────────────────

G, nodes, neighbor_communities = load_graph()
community_centroids = load_community_centroids()
community_labels    = load_community_labels()

# ── Search Index ──────────────────────────────────────────────────────────────

@st.cache_data
def build_search_df():
    df = pd.read_parquet('data/nodes.parquet')[['track_id', 'track_name', 'artists', 'popularity', 'genre']]
    df = df[df['track_name'].apply(lambda x: str(x).isascii())]
    df['label'] = df['track_name'] + ' · ' + df['artists'].str.replace(';', ', ', regex=False)
    # most popular songs appear first in search results
    return df.sort_values('popularity', ascending=False).drop_duplicates(subset='label')

search_df   = build_search_df()
label_to_id = dict(zip(search_df['label'], search_df['track_id']))

# ── Session State ─────────────────────────────────────────────────────────────

for key, default in [
    ('queue',        None),
    ('journey',      None),
    ('breadcrumb',   []),
    ('pending_seed', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Press+Start+2P&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    @media (prefers-color-scheme: dark) {
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], section.main, .main .block-container {
            background-color: #ffffff !important; color: #191414 !important;
        }
    }

    /* hide streamlit anchor link icon */
    a.anchor, .anchor { display: none !important; }
    h1 a, h2 a, h3 a { display: none !important; }

    /* hero */
    .hero { text-align: center; padding: 2.5rem 0 2rem 0; }
    .hero h1 {
        font-family: 'Press Start 2P', monospace;
        font-size: 1.6rem; letter-spacing: 1px; line-height: 1.5;
        margin-bottom: 0.75rem; color: #1db954;
    }
    .hero h1 em { font-style: normal; color: #158a3e; }
    .hero p { color: #888; font-size: 0.95rem; margin: 0; }

    /* generate button */
    div.stButton > button[kind="primary"] {
        background-color: #1db954 !important; border: none !important;
        color: #000 !important; font-weight: 700 !important;
        border-radius: 500px !important; padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important; margin-top: 0.75rem; width: 100%;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #1ed760 !important; }

    /* breadcrumb */
    .breadcrumb {
        display: flex; flex-wrap: wrap; align-items: center;
        gap: 4px; margin: 1.5rem 0 0.25rem 0;
        font-size: 0.75rem; color: #aaa; line-height: 1.6;
    }
    .breadcrumb-song { color: #1db954; font-weight: 600; }
    .breadcrumb-sep  { color: #ccc; font-size: 0.7rem; margin: 0 2px; }

    /* queue header */
    .queue-header {
        font-size: 0.68rem; color: #aaa; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 600;
        margin: 1rem 0 0.4rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }

    /* queue rows */
    .queue-num    { font-size: 0.8rem; color: #bbb; font-weight: 500; min-width: 20px; text-align: right; padding-top: 8px; }
    .queue-track  { font-size: 0.95rem; font-weight: 600; color: #191414; }
    .queue-artist { font-size: 0.82rem; color: #888; margin-top: 1px; }
    .genre-tag {
        display: inline-block; font-size: 0.62rem; color: #999;
        font-weight: 500; background: #f2f2f2; border-radius: 4px;
        padding: 2px 7px; margin-top: 5px;
    }

    /* role pills — square, retro font */
    .role-pill {
        display: inline-block; font-family: 'Press Start 2P', monospace;
        font-size: 0.45rem; text-transform: uppercase;
        padding: 4px 7px; border-radius: 3px; white-space: nowrap;
    }
    .role-start       { background: #e6f9ee; color: #1a7a3c; }
    .role-bridge      { background: #1db954; color: #000; }
    .role-destination { background: #ede9fe; color: #5b21b6; }

    /* branch buttons */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        background: transparent !important; border: none !important;
        color: #bbb !important; font-size: 0.72rem !important;
        font-weight: 500 !important; padding: 0.25rem 0.5rem !important;
        border-radius: 4px !important; width: 100% !important;
        text-align: left !important; cursor: pointer !important;
        white-space: nowrap !important; margin: 0 !important;
    }
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background: #f0faf4 !important; color: #1db954 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero'>
    <h1>Song <em>Path</em> Generator</h1>
    <p>Pick a song, we'll take it from there.</p>
</div>
""", unsafe_allow_html=True)

# handle row-click seed set in a previous rerun
if st.session_state.pending_seed is not None:
    seed_id = st.session_state.pending_seed
    st.session_state.pending_seed = None
    with st.spinner(''):
        try:
            new_queue, journey = generate_queue(
                seed_id, G, nodes, neighbor_communities, community_centroids,
                st.session_state.get('queue_length', 8),
                st.session_state.get('max_artist', 2),
                st.session_state.get('adventurousness', 2),
                st.session_state.get('min_popularity', 35),
            )
            d = G.nodes[seed_id]
            st.session_state.breadcrumb.append((d.get('track_name', ''), d.get('artists', '')))
            st.session_state.queue   = new_queue
            st.session_state.journey = journey
        except ValueError as e:
            st.error(f'Error: {e}')

# search
selection = st.selectbox(
    'Search',
    options=search_df['label'].tolist(),
    index=None,
    placeholder='Search songs or artists...',
    label_visibility='collapsed'
)
selected_track_id = label_to_id.get(selection)

# settings
with st.expander('⚙️ Settings', expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        queue_length = st.slider('Queue length', 5, 20, 8)
    with c2:
        adventurousness = st.slider('Adventurousness', 1, 5, 2,
                                    help='1 = stay close to your genre, 5 = jump far')
    with c3:
        max_artist = st.slider('Max per artist', 1, 3, 2)
    with c4:
        min_popularity = st.slider('Min popularity', 0, 80, 35,
                                   help='Spotify popularity score (0–100). 0 = include everything, 80 = charting hits only')

    st.session_state['queue_length']    = queue_length
    st.session_state['adventurousness'] = adventurousness
    st.session_state['max_artist']      = max_artist
    st.session_state['min_popularity']  = min_popularity

if selected_track_id and st.button('Generate Queue', type='primary', use_container_width=True):
    with st.spinner(''):
        try:
            queue, journey = generate_queue(
                selected_track_id, G, nodes, neighbor_communities, community_centroids,
                queue_length, max_artist, adventurousness, min_popularity
            )
            d = G.nodes[selected_track_id]
            st.session_state.breadcrumb = [(d.get('track_name', ''), d.get('artists', ''))]
            st.session_state.queue   = queue
            st.session_state.journey = journey
        except ValueError as e:
            st.error(f'Error: {e}')

# ── Queue Display ─────────────────────────────────────────────────────────────

if st.session_state.queue is not None:
    queue   = st.session_state.queue
    journey = st.session_state.journey
    crumbs  = st.session_state.breadcrumb

    # breadcrumb trail
    if crumbs:
        parts = []
        for i, (name, _) in enumerate(crumbs):
            if i > 0:
                parts.append("<span class='breadcrumb-sep'>›</span>")
            parts.append(f"<span class='breadcrumb-song'>{html_lib.escape(name)}</span>")
        st.markdown("<div class='breadcrumb'>" + ''.join(parts) + '</div>', unsafe_allow_html=True)

    if len(crumbs) > 1 and st.button('↩ Start over', key='reset'):
        st.session_state.queue      = None
        st.session_state.journey    = None
        st.session_state.breadcrumb = []
        st.rerun()


    # seed song audio profile
    seed_id = queue.iloc[0]['track_id'] if len(queue) > 0 else None
    if seed_id and seed_id in G.nodes:
        sd = G.nodes[seed_id]
        def pct(val):
            return f'{int(float(val) * 100)}%' if val is not None else '—'
        def bpm(val):
            return f'{int(float(val))} bpm' if val is not None else '—'
        st.markdown(f"""
        <div style='font-size:0.68rem;color:#aaa;margin:0.5rem 0 1rem 0;
                    display:flex;flex-wrap:wrap;gap:16px;align-items:center'>
            <span style='color:#bbb;font-weight:600;text-transform:uppercase;
                         letter-spacing:0.1em;font-size:0.6rem'>seed song profile</span>
            <span>energy <strong style='color:#555'>{pct(sd.get('energy'))}</strong></span>
            <span>valence <strong style='color:#555'>{pct(sd.get('valence'))}</strong></span>
            <span>danceability <strong style='color:#555'>{pct(sd.get('danceability'))}</strong></span>
            <span>tempo <strong style='color:#555'>{bpm(sd.get('tempo'))}</strong></span>
            <span>acousticness <strong style='color:#555'>{pct(sd.get('acousticness'))}</strong></span>
        </div>""", unsafe_allow_html=True)


    st.markdown(f"<div class='queue-header'>{len(queue)} tracks · click → to branch from any song</div>",
                unsafe_allow_html=True)

    # queue rows
    for i, row in queue.iterrows():
        col_num, col_info, col_role, col_btn = st.columns([0.5, 7, 2, 1.5])

        track_name = html_lib.escape(str(row['track_name'] or ''))
        artist     = html_lib.escape(str(row['artist'] or '').replace(';', ', '))
        genre      = html_lib.escape(str(row['genre']  or ''))

        role_html = {
            'start':       "<span class='role-pill role-start'>Start</span>",
            'bridge':      "<span class='role-pill role-bridge'>Bridge</span>",
            'destination': "<span class='role-pill role-destination'>Destination</span>",
        }.get(row.get('role'), '')

        with col_num:
            st.markdown(f"<div class='queue-num'>{i + 1}</div>", unsafe_allow_html=True)
        with col_info:
            st.markdown(f"""
            <div style='padding:6px 0'>
                <div class='queue-track'>{track_name}</div>
                <div class='queue-artist'>{artist}</div>
                <span class='genre-tag'>{genre}</span>
            </div>""", unsafe_allow_html=True)
        with col_role:
            st.markdown(f"<div style='padding-top:10px'>{role_html}</div>", unsafe_allow_html=True)
        with col_btn:
            if st.button('→', key=f"branch_{i}_{row['track_id']}"):
                st.session_state.pending_seed = row['track_id']
                st.rerun()
