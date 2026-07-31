import pandas as pd
from ast import literal_eval
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "": "#",
    "ECOLO": "#",
    "HOR": "#",
    "LIOT": "#",
    "RE": "#",
    "RN": "#",
    "SOC": "#",
    "DEM": "#",
    "GDR-NUPES": "#",
    "LFI-NUPES": "#",
    "LR": "#",
}

df = pd.read_csv(
    "./data/embed_anti-squat.csv", converters={"embedding_speech": literal_eval}
)

amds = pd.read_csv(
    "./data/embed_amendments_anti_squat.csv",
    converters={"embedding_amendment_summary": literal_eval},
)
amds = amds.dropna(subset=["embedding_amendment_summary"])
amds = amds.to_dict(orient="records")
amds = {a["number"]: a for a in amds}


def treat_debate(debate):
    for interv in debate:
        nums = interv["amendments"].split("|")
        if nums == [""]:
            interv["embedding_normalized"] = interv["embedding_speech"]
            continue

        # On récupère les embeddings des exposés sommaires
        # des amendements concernés par les interventions du débat
        umds = [amds[n] for n in nums if n in amds]
        umds = [m["embedding_amendment_summary"] for m in umds]
        if not umds:
            interv["embedding_normalized"] = interv["embedding_speech"]
            continue
        umds = np.array(umds)
        umds = umds.mean(axis=0)

        embed = interv["embedding_speech"]
        embed = np.array(embed)

        amd_dir = umds / np.linalg.norm(umds)  # direction unitaire des amendements
        embed = embed - np.dot(embed, amd_dir) * amd_dir

        interv["embedding_normalized"] = embed.tolist()

    return debate


corpus = []

debates = df.groupby("debate")
for debate, groups in debates:
    groups = groups.fillna("").to_dict(orient="records")
    groups = treat_debate(groups)
    corpus.append(groups)


# VISUALIZATION

for item in corpus:
    embeddings = np.array([i["embedding_speech"] for i in item])
    normalized = np.array([i["embedding_normalized"] for i in item])
    texts = [i["speech"] for i in item]
    subject = [i["speech_subject_2"] for i in item][0]

    groups = [i["speaker_group"] for i in item]

    if len(embeddings) < 2:
        continue

    # PCA
    pca_a = PCA(n_components=2)
    pca_embeddings = pca_a.fit_transform(embeddings)

    pca_b = PCA(n_components=2)
    pca_normalized = pca_b.fit_transform(normalized)

    # Checking PCA differences

    diff = embeddings - normalized
    print(f"Debate {item[0]['debate']}")
    print(f"max|raw-normalized| = {np.max(np.abs(diff)):.3e}")
    print(
        f"allclose(raw, normalized) = {np.allclose(embeddings, normalized, atol=1e-10)}"
    )
    print(f"PCA A variance ratio = {pca_a.explained_variance_ratio_}")
    print(f"PCA B variance ratio = {pca_b.explained_variance_ratio_}")
    # Variance de la correction dans chaque composante principale de A
    diff_in_pca_space = pca_a.transform(normalized) - pca_embeddings
    print(
        f"Variance correction dans PC1={np.var(diff_in_pca_space[:, 0]):.3e}, PC2={np.var(diff_in_pca_space[:, 1]):.3e}"
    )
    print(f"Variance correction totale={np.var(diff):.3e}")
    print("-" * 60)

    # Color mapping by speaker group
    unique_groups = sorted(set(groups))
    cmap = plt.cm.get_cmap("tab10", len(unique_groups))
    group_to_color = {g: cmap(ix) for ix, g in enumerate(unique_groups)}
    colors = [group_to_color[g] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(subject)

    sc1 = axes[0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=colors, s=45)
    axes[0].set_title("Embeddings (PCA)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sc2 = axes[1].scatter(pca_normalized[:, 0], pca_normalized[:, 1], c=colors, s=45)
    axes[1].set_title("Normalized embeddings (PCA)")
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
    fig.legend(handles=handles, loc="upper right", ncol=min(len(unique_groups), 7))

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
        for sc, coords in [(sc1, pca_embeddings), (sc2, pca_normalized)]:
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
