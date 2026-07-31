import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
from ast import literal_eval
from sklearn.decomposition import PCA
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche sémantique — Débats", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 0.75rem !important; padding-bottom: 0 !important; }
    h1 { margin-bottom: 0.25rem !important; }
    h3 { margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)
st.title("Recherche sémantique — Débats 17e législature")

CSV_1 = "./corpus/post-train/amendements_finetuned_embed.csv"
CSV_2 = "./corpus/pre-train/amendements_qwen_embed.csv"
MODEL_PATH_1 = "./model-lora2"
MODEL_PATH_2 = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_DIR = ".chroma"
BATCH_SIZE = 500
DISPLAY_COLS = ["distance", "author_group", "dossier", "amendment_summary"]

GROUP_COLORS = {
    "": "#05B2F6",
    "DR": "#3B79EC",
    "EPR": "#05B2F6",
    "HOR": "#05B2F6",
    "LIOT": "#6C6A6A",
    "RN": "#4407ED",
    "UDDPLR": "#4407ED",
    "DEM": "#05B2F6",
    "UDR": "#05B2F6",
    "ECOS": "#3BD813",
    "GDR": "#E01111",
    "LFI-NFP": "#E01111",
    "NI": "#6C6A6A",
    "SOC": "#EB11CE",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Nombre de résultats", min_value=1, max_value=50, value=10)


# ── Data loading & indexing ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de l'index Chroma…")
def build_chroma_index():
    """Open (or create) persistent Chroma collections; upsert CSVs only if empty."""

    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["embedding_amendment_summary"] = df["embedding_amendment_summary"].apply(literal_eval)
        df["author_group"] = df["author_group"].fillna("").astype(str)
        df["amendment_id"] = [str(i) for i in range(len(df))]
        return df

    def upsert_collection(collection, df: pd.DataFrame) -> None:
        for start in range(0, len(df), BATCH_SIZE):
            sub = df.iloc[start : start + BATCH_SIZE]
            collection.upsert(
                ids=sub["amendment_id"].tolist(),
                embeddings=sub["embedding_amendment_summary"].tolist(),
                documents=sub["amendment_summary"].tolist(),
                metadatas=[
                    {
                        "dossier": row["dossier"],
                        "author_group": row["author_group"],
                    }
                    for _, row in sub.iterrows()
                ],
            )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col1 = client.get_or_create_collection("model_17th", metadata={"hnsw:space": "cosine"})
    col2 = client.get_or_create_collection("model_qwen", metadata={"hnsw:space": "cosine"})

    if col1.count() == 0:
        upsert_collection(col1, load_csv(CSV_1))
    if col2.count() == 0:
        upsert_collection(col2, load_csv(CSV_2))

    return col1, col2


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models():
    model1 = SentenceTransformer(MODEL_PATH_1)
    model2 = SentenceTransformer(MODEL_PATH_2)
    return model1, model2


def make_pca_plot(
    df_results: pd.DataFrame,
    result_embeddings: list,
    query_emb: list[float],
) -> "px.Figure":
    """Fit PCA on the search results + query point and plot."""
    matrix = np.vstack([result_embeddings, query_emb]).astype(np.float32)
    coords = PCA(n_components=2).fit_transform(matrix)
    n = len(result_embeddings)

    df = df_results[["author_group", "dossier"]].copy()
    df["pc1"] = coords[:n, 0]
    df["pc2"] = coords[:n, 1]

    fig = px.scatter(
        df,
        x="pc1",
        y="pc2",
        color="speaker_group",
        color_discrete_map=GROUP_COLORS,
        hover_data={"dossier": True, "pc1": False, "pc2": False},
        opacity=0.85,
        size_max=10,
        labels={"pc1": "PC1", "pc2": "PC2", "author_group": "Groupe"},
    )

    fig.add_scatter(
        x=[coords[n, 0]],
        y=[coords[n, 1]],
        mode="markers",
        marker=dict(size=16, symbol="x", color="black", line=dict(width=2, color="black")),
        name="Requête",
        hovertemplate="<b>Requête</b><extra></extra>",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        height=300,
    )
    return fig


def search(
    collection, query_embedding: list[float], k: int
) -> tuple[pd.DataFrame, list]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["embeddings", "documents", "metadatas", "distances"],
    )
    rows = []
    for id_, dist, doc, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        rows.append(
            {
                "amendment_id": id_,
                "distance": round(float(dist), 4),
                "amendment_summary": doc,
                "dossier": meta.get("dossier", ""),
                "author_group": meta.get("author_group", ""),
            }
        )
    df = pd.DataFrame(rows)
    return df[DISPLAY_COLS], results["embeddings"][0]


# ── Main UI ────────────────────────────────────────────────────────────────────
col1_idx, col2_idx = build_chroma_index()
encoder1, encoder2 = load_models()

query_text = st.text_input("Requête", placeholder="Entrez votre requête…")

if not query_text:
    st.stop()

with st.spinner("Encodage de la requête…"):
    embedding1 = encoder1.encode([query_text])[0].tolist()
    embedding2 = encoder2.encode([query_text])[0].tolist()

df_res1, emb1 = search(col1_idx, embedding1, top_k)
df_res2, emb2 = search(col2_idx, embedding2, top_k)

col_left, col_right = st.columns(2)

COLUMN_CONFIG = {
    "distance": st.column_config.NumberColumn("Distance", format="%.4f", width="small"),
    "author_group": st.column_config.TextColumn("Groupe", width="small"),
    "dossier": st.column_config.TextColumn("Dossier", width="medium"),
    "amendment_summary": st.column_config.TextColumn("Contenu", width="large"),
}

with col_left:
    st.subheader("Pre-trained")
    st.plotly_chart(make_pca_plot(df_res2, emb2, embedding2), use_container_width=True)
    st.dataframe(
        df_res2,
        use_container_width=True,
        column_config=COLUMN_CONFIG,
        hide_index=True,
        height=500,
    )

with col_right:
    st.subheader("Post-trained")
    st.plotly_chart(make_pca_plot(df_res1, emb1, embedding1), use_container_width=True)
    st.dataframe(
        df_res1,
        use_container_width=True,
        column_config=COLUMN_CONFIG,
        hide_index=True,
        height=500,
    )
