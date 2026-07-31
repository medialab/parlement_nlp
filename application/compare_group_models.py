import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amendements import (
	COLORS,
	build_topic_metadata,
	compute_top_k_neighbors,
	compute_topic_hulls,
	fit_topic_model,
	parse_embedding,
	pca_2d,
)


def load_group_dataframe(csv_path, group):
	df = pd.read_csv(
		csv_path,
		converters={"embedding_amendment_summary": parse_embedding},
	)

	required_cols = {"number", "author_group", "amendment_summary", "embedding_amendment_summary"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

	df = df.dropna(subset=["amendment_summary", "embedding_amendment_summary"]).copy()
	df["author_group"] = df["author_group"].fillna("UNKNOWN").astype(str)
	df = df[df["author_group"] == group].reset_index(drop=True)

	if len(df) < 2:
		raise ValueError(f"Need at least 2 amendments for group '{group}' in {csv_path}, found {len(df)}")

	return df


def prepare_model_view(csv_path, group, k, min_topic_size):
	df = load_group_dataframe(csv_path, group)
	embeddings = np.array(df["embedding_amendment_summary"].tolist(), dtype=float)
	neighbors_idx, neighbors_sim = compute_top_k_neighbors(embeddings, k)
	coords = pca_2d(embeddings)

	effective_min_topic_size = min(min_topic_size, max(2, len(df) // 2))
	model, topics = fit_topic_model(
		embeddings,
		df["amendment_summary"].tolist(),
		min_topic_size=effective_min_topic_size,
	)
	df["topic"] = topics
	topic_meta_df = build_topic_metadata(df, topics, model)

	return {
		"df": df,
		"coords": coords,
		"neighbors_idx": neighbors_idx,
		"neighbors_sim": neighbors_sim,
		"topic_meta_df": topic_meta_df,
		"topics": topics,
		"sample_size": len(df),
		"effective_min_topic_size": effective_min_topic_size,
	}


def build_keyword_rows(label, topic_meta_df):
	if topic_meta_df.empty:
		return [{"model": label, "topic": "-", "size": 0, "share": "0.0%", "keywords": "No topic found"}]

	rows = []
	for row in topic_meta_df.sort_values(["size", "topic"], ascending=[False, True]).itertuples(index=False):
		rows.append(
			{
				"model": label,
				"topic": str(int(row.topic)),
				"size": int(row.size),
				"share": f"{row.share:.1%}",
				"keywords": row.top_words,
			}
		)
	return rows


def make_topic_hover_text(topic, meta):
	if meta is None:
		return f"Topic {topic}"
	return (
		f"Topic {topic}"
		f"<br>Size: {meta.size}"
		f"<br>Share: {meta.share:.1%}"
		f"<br>Dominant group: {meta.dominant_group} ({meta.dominant_group_share:.1%})"
		f"<br>Keywords: {meta.top_words}"
	)


def add_model_panel(fig, panel, model_key, model_label, group, color, k, col):
	coords = panel["coords"]
	df = panel["df"]
	topics = panel["topics"]
	topic_meta_df = panel["topic_meta_df"]

	x = coords[:, 0]
	y = coords[:, 1]
	topic_palette = px.colors.qualitative.Pastel + px.colors.qualitative.Light24 + px.colors.qualitative.Set3
	topic_meta = {}
	if not topic_meta_df.empty:
		topic_meta = {int(row.topic): row for row in topic_meta_df.itertuples(index=False)}

	topic_colors = {-1: "#9ca3af"}
	for idx, topic in enumerate(sorted(np.unique(topics))):
		if topic == -1:
			continue
		topic_colors[int(topic)] = topic_palette[idx % len(topic_palette)]

	hulls = compute_topic_hulls(coords, topics)
	for topic, hull_x, hull_y in hulls:
		meta = topic_meta.get(int(topic))
		keywords = meta.top_words if meta is not None else "N/A"
		legend_keywords = ", ".join(keywords.split(", ")[:3]) if keywords else "N/A"
		fig.add_trace(
			go.Scatter(
				x=hull_x,
				y=hull_y,
				mode="lines",
				name=f"{model_label} T{topic} | {legend_keywords}",
				legendgroup=model_key,
				legendgrouptitle_text=model_label,
				line={"width": 0, "color": topic_colors.get(int(topic), color)},
				fill="toself",
				fillcolor=topic_colors.get(int(topic), color),
				opacity=0.18,
				text=[make_topic_hover_text(topic, meta)] * len(hull_x),
				hovertemplate="%{text}<extra></extra>",
				showlegend=True,
				meta={"trace_type": "topic_hull", "model": model_key, "topic": int(topic)},
			),
			row=1,
			col=col,
		)

	for topic in sorted(np.unique(topics)):
		mask = topics == topic
		if not np.any(mask):
			continue
		hover_text = []
		idxs = np.where(mask)[0]
		for idx in idxs:
			summary = str(df.iloc[idx]["amendment_summary"])
			summary_short = summary[:260] + "..." if len(summary) > 260 else summary
			hover_text.append(
				f"number={df.iloc[idx]['number']}"
				f"<br>group={group}"
				f"<br>topic={int(topic)}"
				f"<br>{summary_short}"
			)

		fig.add_trace(
			go.Scattergl(
				x=x[idxs],
				y=y[idxs],
				mode="markers",
				name=f"{model_label} points T{int(topic)}",
				legendgroup=model_key,
				showlegend=False,
				customdata=idxs,
				text=hover_text,
				hovertemplate="%{text}<extra></extra>",
				marker={
					"size": 8,
					"opacity": 0.9,
					"color": topic_colors.get(int(topic), color),
					"line": {"width": 0.6, "color": "#111827"},
				},
				meta={"trace_type": "amendment_point", "model": model_key},
			),
			row=1,
			col=col,
		)

	fig.add_trace(
		go.Scatter(
			x=[],
			y=[],
			mode="lines",
			line={"width": 1.3, "color": "#374151"},
			hoverinfo="skip",
			showlegend=False,
			meta={"trace_type": "neighbor_edges", "model": model_key},
		),
		row=1,
		col=col,
	)
	edge_trace_idx = len(fig.data) - 1

	fig.add_trace(
		go.Scatter(
			x=[],
			y=[],
			mode="markers",
			marker={"size": 14, "symbol": "x", "color": "#111111"},
			hoverinfo="skip",
			showlegend=False,
			meta={"trace_type": "selected_point", "model": model_key},
		),
		row=1,
		col=col,
	)
	selected_trace_idx = len(fig.data) - 1

	fig.update_xaxes(title_text=f"Component 1 · top {k} voisins au clic", row=1, col=col)
	fig.update_yaxes(title_text="Component 2", row=1, col=col)

	return {
		"edge_trace_idx": edge_trace_idx,
		"selected_trace_idx": selected_trace_idx,
		"xs": x.tolist(),
		"ys": y.tolist(),
		"neighbors": panel["neighbors_idx"].tolist(),
	}


def build_comparison_figure(pre_panel, post_panel, group, dataset_name, k):
	group_color = COLORS.get(group, "#d62728")
	fig = make_subplots(
		rows=1,
		cols=3,
		specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "table"}]],
		column_widths=[0.37, 0.37, 0.26],
		horizontal_spacing=0.04,
		subplot_titles=(
			f"Pre-train · {group}",
			f"Post-train · {group}",
			"Topics et mots-clés",
		),
	)

	pre_js = add_model_panel(fig, pre_panel, "pre", "Pre", group, group_color, k, col=1)
	post_js = add_model_panel(fig, post_panel, "post", "Post", group, group_color, k, col=2)

	legend_rows = build_keyword_rows("Pre", pre_panel["topic_meta_df"]) + build_keyword_rows("Post", post_panel["topic_meta_df"])
	row_fill_colors = ["#ffffff" if row["model"] == "Pre" else "#f9fafb" for row in legend_rows]
	fig.add_trace(
		go.Table(
			header={
				"values": ["Modèle", "Topic", "Taille", "Part", "Mots-clés"],
				"align": "left",
				"fill_color": "#e5e7eb",
				"font": {"size": 12, "color": "#111827"},
			},
			cells={
				"values": [
					[row["model"] for row in legend_rows],
					[row["topic"] for row in legend_rows],
					[row["size"] for row in legend_rows],
					[row["share"] for row in legend_rows],
					[row["keywords"] for row in legend_rows],
				],
				"align": "left",
				"fill_color": [row_fill_colors] * 5,
				"font": {"size": 11, "color": "#111827"},
				"height": 26,
			},
		),
		row=1,
		col=3,
	)

	fig.add_annotation(
		x=0.5,
		y=1.09,
		xref="paper",
		yref="paper",
		showarrow=False,
		text=(
			f"{dataset_name} · groupe {group}"
			f"<br><sup>Survolez un halo pour les détails de topic, survolez un point pour l'amendement, cliquez pour afficher les {k} voisins les plus proches.</sup>"
		),
		font={"size": 14, "color": "#111827"},
	)

	fig.update_layout(
		template="plotly_white",
		title=f"Comparaison de clusterisation BERTopic · {dataset_name}",
		width=1800,
		height=900,
		margin={"l": 40, "r": 20, "t": 120, "b": 40},
		legend={
			"title": {"text": "Halos de topics + mots-clés"},
			"x": 1.01,
			"xanchor": "left",
			"y": 1.0,
			"yanchor": "top",
			"groupclick": "toggleitem",
		},
	)

	post_script = "const gd = document.getElementById('{plot_id}');\n"
	post_script += f"const panels = {json.dumps({'pre': pre_js, 'post': post_js})};\n"
	post_script += """
gd.on('plotly_click', function(evt) {
	if (!evt || !evt.points || evt.points.length === 0) return;
	const point = evt.points[0];
	if (!point.data || !point.data.meta || point.data.meta.trace_type !== 'amendment_point') return;

	const model = point.data.meta.model;
	const panel = panels[model];
	if (!panel) return;

	const idx = Number(point.customdata);
	if (Number.isNaN(idx)) return;

	const edgeX = [];
	const edgeY = [];
	for (let n = 0; n < panel.neighbors[idx].length; n++) {
		const j = panel.neighbors[idx][n];
		edgeX.push(panel.xs[idx], panel.xs[j], null);
		edgeY.push(panel.ys[idx], panel.ys[j], null);
	}

	Plotly.restyle(gd, {x: [edgeX], y: [edgeY]}, [panel.edge_trace_idx]);
	Plotly.restyle(gd, {x: [[panel.xs[idx]]], y: [[panel.ys[idx]]]}, [panel.selected_trace_idx]);
});
"""

	return fig, post_script


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pre-csv",
		default="corpus/pre-train/mayotte_qwen_embed.csv",
		help="CSV pré-train contenant les embeddings des amendements",
	)
	parser.add_argument(
		"--post-csv",
		default="corpus/post-train/mayotte_finetuned_embed.csv",
		help="CSV post-train contenant les embeddings des amendements",
	)
	parser.add_argument("--group", default="LFI-NFP", help="Groupe parlementaire à filtrer")
	parser.add_argument("--dataset-name", default="mayotte", help="Nom affiché du corpus comparé")
	parser.add_argument("--k", type=int, default=5, help="Nombre de voisins au clic")
	parser.add_argument("--min-topic-size", type=int, default=5, help="Taille minimale BERTopic")
	parser.add_argument(
		"--html-out",
		default="corpus/comparison/mayotte_lfi_nfp_pre_post_topics.html",
		help="Sortie HTML de comparaison interactive",
	)
	parser.add_argument(
		"--topics-out-prefix",
		default="corpus/comparison/mayotte_lfi_nfp",
		help="Préfixe des CSV exportés pour les topics pré et post",
	)
	args = parser.parse_args()

	if args.k <= 0:
		raise ValueError("k must be strictly positive")

	pre_csv = Path(args.pre_csv)
	post_csv = Path(args.post_csv)
	if not pre_csv.exists():
		raise FileNotFoundError(f"Pre-train CSV not found: {pre_csv}")
	if not post_csv.exists():
		raise FileNotFoundError(f"Post-train CSV not found: {post_csv}")

	pre_panel = prepare_model_view(pre_csv, args.group, args.k, args.min_topic_size)
	post_panel = prepare_model_view(post_csv, args.group, args.k, args.min_topic_size)

	prefix = Path(args.topics_out_prefix)
	prefix.parent.mkdir(parents=True, exist_ok=True)
	for suffix, panel in (("pre", pre_panel), ("post", post_panel)):
		topic_export = panel["df"][["number", "author_group", "amendment_summary", "topic"]].copy()
		topic_export = topic_export.merge(panel["topic_meta_df"], on="topic", how="left")
		topic_export.to_csv(prefix.parent / f"{prefix.name}_{suffix}_topics.csv", index=False)

	fig, post_script = build_comparison_figure(pre_panel, post_panel, args.group, args.dataset_name, args.k)
	html_out = Path(args.html_out)
	html_out.parent.mkdir(parents=True, exist_ok=True)
	fig.write_html(str(html_out), include_plotlyjs="cdn", post_script=post_script)

	print(f"Saved comparison HTML: {html_out}")
	print(f"Pre-train sample size for {args.group}: {pre_panel['sample_size']}")
	print(f"Post-train sample size for {args.group}: {post_panel['sample_size']}")
	print(f"Pre-train topics: {len(np.unique(pre_panel['topics'][pre_panel['topics'] != -1]))}")
	print(f"Post-train topics: {len(np.unique(post_panel['topics'][post_panel['topics'] != -1]))}")


if __name__ == "__main__":
	main()