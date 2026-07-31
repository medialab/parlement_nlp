import pandas as pd
from ast import literal_eval
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
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


print("Opening file...")
df_finetuned = pd.read_csv(
    "./data/embed_17th_disc_gen.csv", converters={"embedding_speech": literal_eval}
)
df_qwen = pd.read_csv(
    "./data/embed_qwen_17th_disc_gen.csv", converters={"embedding_speech": literal_eval}
)


print("Treating embeddings...")
corpus_finetuned = []
debates_finetuned = df_finetuned.groupby(["speech_subject_1", "speech_date"])
for (debate, date), groups in debates_finetuned:
    groups = groups.fillna("").to_dict(orient="records")
    corpus_finetuned.append(groups)

corpus_qwen = []
debates_qwen = df_qwen.groupby(["speech_subject_1", "speech_date"])
for (debate, date), groups in debates_qwen:
    groups = groups.fillna("").to_dict(orient="records")
    corpus_qwen.append(groups)

# VISUALIZATION

for item_finetuned, item_qwen in zip(corpus_finetuned, corpus_qwen):
    qwen = np.array([i["embedding_speech"] for i in item_qwen])
    embeddings = np.array([i["embedding_speech"] for i in item_finetuned])
    texts = [i["speech"] for i in item_finetuned]
    subject = [i["speech_subject_1"] for i in item_finetuned][0]
    date = [i["speech_date"] for i in item_finetuned][0]

    groups = [i["speaker_group"] for i in item_finetuned]

    if len(embeddings) < 2:
        continue

    # PCA 
    # TODO vérifier d'utiliser 2 PCA plutôt qu'une seule fit sur 
    # un ensemble de vecteurs
    pca_a = PCA(n_components=2)
    pca_qwen = pca_a.fit_transform(qwen)

    pca_b = PCA(n_components=2)
    pca_embeddings = pca_b.fit_transform(embeddings)

    # Checking PCA differences

    # Color mapping by speaker group
    unique_groups = sorted(set(groups))
    cmap = plt.cm.get_cmap("tab10", len(unique_groups))
    group_to_color = COLORS
    colors = [group_to_color[g] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    fig.suptitle(f"{subject} - {date}")

    sc1 = axes[0].scatter(pca_qwen[:, 0], pca_qwen[:, 1], c=colors, s=45)
    axes[0].set_title("Embeddings (PCA)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sc2 = axes[1].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=colors, s=45)
    axes[1].set_title("Embeddings (PCA)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")


    # Shared legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(g),
            markerfacecolor=group_to_color[g],
            markersize=8,
        )
        for g in unique_groups
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(unique_groups), 7),
    )

    # Tooltip on hover
    tooltip = fig.text(
        0.5,
        0.01,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9),
    )

    def on_hover(event):
        for sc in [sc1, sc2]:
            if event.inaxes != sc.axes:
                continue
            cont, ind = sc.contains(event)
            if cont:
                idx = ind["ind"][0]
                raw = texts[idx]
                snippet = raw[:300] + ("…" if len(raw) > 300 else "")
                tooltip.set_text(snippet)
                fig.canvas.draw_idle()
                return
        tooltip.set_text("")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
