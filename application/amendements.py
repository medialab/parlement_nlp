import argparse
import html
import json
import textwrap
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bertopic import BERTopic
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer

COLORS = {
	"RN": "#01159C",
    "SOC": "#FF00B7",
    "LIOT": "#636363",
    "EPR": "#6C88A5",
    "UDDPLR": "#000399",
    "HOR": "#6667A1",
    "GDR": "#007BFF",
    "ECOS": "#009207",
    "DR": "#003CFF",
    "DEM": "#506080",
    "NI": "#6D6D6D",
    "LFI-NFP": "#FF0000",
    "UDR": "#6C7A9C",
}


def parse_embedding(value):
	if isinstance(value, list):
		return value
	if isinstance(value, str):
		return literal_eval(value)
	raise ValueError("Unsupported embedding format")


def compute_top_k_neighbors(embeddings, k):
	n_samples = embeddings.shape[0]
	k = min(k, max(0, n_samples - 1))
	if k == 0:
		return np.empty((n_samples, 0), dtype=int), np.empty((n_samples, 0), dtype=float)

	norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
	norms = np.where(norms == 0, 1.0, norms)
	normalized = embeddings / norms

	neighbors_idx = np.zeros((n_samples, k), dtype=int)
	neighbors_sim = np.zeros((n_samples, k), dtype=float)

	for i in range(n_samples):
		sims = normalized @ normalized[i]
		sims[i] = -np.inf
		idx = np.argpartition(-sims, kth=k - 1)[:k]
		idx = idx[np.argsort(-sims[idx])]
		neighbors_idx[i] = idx
		neighbors_sim[i] = sims[idx]

	return neighbors_idx, neighbors_sim


def pca_2d(embeddings):
	x = embeddings - embeddings.mean(axis=0, keepdims=True)
	if x.shape[1] == 1:
		return np.hstack([x, np.zeros((x.shape[0], 1), dtype=x.dtype)])

	_, _, vt = np.linalg.svd(x, full_matrices=False)
	components = vt[:2]
	return x @ components.T


def fit_topic_model(embeddings, texts, min_topic_size=5):
	"""Fit BERTopic using pre-computed embeddings."""
	vectorizer = CountVectorizer(min_df=0.2, max_df=0.8)
	model = BERTopic(min_topic_size=min_topic_size, embedding_model=None, verbose=False, vectorizer_model=vectorizer)
	topics, probs = model.fit_transform(texts, embeddings=embeddings)
	return model, topics


def compute_topic_hulls(coords, topics):
	"""Compute convex hull outlines for each topic cluster."""
	from scipy.spatial import ConvexHull

	hulls = []
	for topic in sorted(np.unique(topics)):
		mask = topics == topic
		if mask.sum() < 3:
			continue
		pts = coords[mask]
		try:
			hull = ConvexHull(pts)
			hull_pts = pts[hull.vertices]
			hull_pts = np.vstack([hull_pts, hull_pts[0]])
			hulls.append((topic, hull_pts[:, 0], hull_pts[:, 1]))
		except Exception:
			pass
	return hulls


def build_topic_metadata(df, topics, topic_model):
	"""Build topic-level metadata for hover and reporting."""
	topic_series = pd.Series(topics, name="topic")
	df_local = df.copy()
	df_local["topic"] = topic_series
	total = len(df_local)

	rows = []
	for topic in sorted(df_local["topic"].dropna().unique().tolist()):
		if topic == -1:
			continue
		topic_df = df_local[df_local["topic"] == topic]
		size = len(topic_df)
		share = (size / total) if total else 0.0

		group_counts = topic_df["author_group"].fillna("UNKNOWN").astype(str).value_counts()
		dominant_group = group_counts.index[0] if len(group_counts) else "UNKNOWN"
		dominant_share = (group_counts.iloc[0] / size) if size else 0.0
		top_groups = group_counts.index[:6].tolist()
		top_group_counts = group_counts.iloc[:6].astype(int).tolist()
		top_group_shares = ((group_counts.iloc[:6] / size) if size else pd.Series(dtype=float)).tolist()

		topic_words = topic_model.get_topic(topic) or []
		top_words = [word for word, _ in topic_words[:6]]
		top_words_text = ", ".join(top_words) if top_words else "N/A"

		rows.append(
			{
				"topic": int(topic),
				"size": int(size),
				"share": float(share),
				"dominant_group": dominant_group,
				"dominant_group_share": float(dominant_share),
				"top_words": top_words_text,
				"top_groups": top_groups,
				"top_group_counts": top_group_counts,
				"top_group_shares": top_group_shares,
			}
		)

	meta_df = pd.DataFrame(rows)
	if meta_df.empty:
		return meta_df
	return meta_df.sort_values(["size", "topic"], ascending=[False, True]).reset_index(drop=True)


def save_topic_cluster_overview(topic_meta_df, html_path):
	"""Save a Plotly chart that summarizes topic clusters for plfss."""
	if topic_meta_df.empty:
		fig = go.Figure()
		fig.add_annotation(
			text="No BERTopic clusters found (all points assigned to outlier topic -1).",
			x=0.5,
			y=0.5,
			xref="paper",
			yref="paper",
			showarrow=False,
		)
		fig.update_layout(template="plotly_white", width=1100, height=500, title="PLFSS Topic Cluster Overview")
		fig.write_html(str(html_path), include_plotlyjs="cdn")
		return

	plot_df = topic_meta_df.sort_values("size", ascending=False)
	hover_text = [
		(
			f"Topic {int(row.topic)}"
			f"<br>Size: {int(row.size)}"
			f"<br>Share: {row.share:.1%}"
			f"<br>Dominant group: {row.dominant_group} ({row.dominant_group_share:.1%})"
			f"<br>Keywords: {row.top_words}"
		)
		for row in plot_df.itertuples(index=False)
	]

	fig = go.Figure(
		data=go.Bar(
			x=plot_df["topic"].astype(str),
			y=plot_df["size"],
			marker={
				"color": plot_df["dominant_group_share"],
				"colorscale": "Teal",
				"colorbar": {"title": "dominant group share"},
			},
			hovertext=hover_text,
			hovertemplate="%{hovertext}<extra></extra>",
		)
	)

	fig.update_layout(
		template="plotly_white",
		title="PLFSS Topic Clusters: size, dominant group, and keywords",
		xaxis_title="Topic",
		yaxis_title="Number of amendments",
		width=1200,
		height=620,
		margin={"l": 60, "r": 60, "t": 70, "b": 60},
	)

	fig.write_html(str(html_path), include_plotlyjs="cdn")


def build_neighbors_dataframe(df, neighbors_idx, neighbors_sim):
	rows = []
	for i in range(len(df)):
		for rank, (j, sim) in enumerate(zip(neighbors_idx[i], neighbors_sim[i]), start=1):
			rows.append(
				{
					"amendment_idx": i,
					"amendment_number": df.iloc[i]["number"],
					"author_group": df.iloc[i]["author_group"],
					"neighbor_rank": rank,
					"neighbor_idx": int(j),
					"neighbor_number": df.iloc[j]["number"],
					"neighbor_author_group": df.iloc[j]["author_group"],
					"cosine_similarity": float(sim),
				}
			)
	return pd.DataFrame(rows)


def build_group_topic_cluster_matrices(df):
	"""Build group-to-group matrices from BERTopic clusters using R_union.

	For groups g1 and g2 with topic sets T_g1 and T_g2 (excluding topic -1):
	share[g1, g2] = |T_g1 ∩ T_g2| / |T_g1 ∪ T_g2|
	"""
	if "topic" not in df.columns:
		raise ValueError("Column 'topic' is required to build topic cluster matrices")

	clean = df.copy()
	clean["author_group"] = clean["author_group"].fillna("UNKNOWN").astype(str)
	all_groups = sorted(clean["author_group"].unique().tolist())
	clean = clean[clean["topic"] != -1]

	topic_counts = pd.crosstab(clean["author_group"], clean["topic"]).sort_index()
	if topic_counts.empty:
		empty_counts = pd.DataFrame(0, index=all_groups, columns=all_groups, dtype=int)
		empty_share = pd.DataFrame(0.0, index=all_groups, columns=all_groups, dtype=float)
		return empty_counts, empty_share

	presence = (topic_counts > 0).astype(int)
	presence = presence.reindex(all_groups, fill_value=0)

	intersection = presence @ presence.T
	topic_cardinality = presence.sum(axis=1)
	union = pd.DataFrame(
		index=presence.index,
		columns=presence.index,
		dtype=float,
	)
	for g1 in presence.index:
		for g2 in presence.index:
			union.loc[g1, g2] = float(topic_cardinality[g1] + topic_cardinality[g2] - intersection.loc[g1, g2])

	share = intersection.astype(float).div(union.replace(0.0, np.nan)).fillna(0.0)
	share = share.sort_index().sort_index(axis=1)

	counts = intersection.reindex(index=share.index, columns=share.columns).fillna(0).astype(int)

	return counts, share


def save_group_neighbor_heatmap(percentages, html_path):
	fig = go.Figure(
		data=go.Heatmap(
			z=percentages.values,
			x=percentages.columns.tolist(),
			y=percentages.index.tolist(),
			colorscale="Blues",
			zmin=0,
			zmax=1,
			colorbar={"title": "R_union"},
			text=np.round(percentages.values * 100, 1),
			texttemplate="%{text}%",
			hovertemplate="group_a=%{y}<br>group_b=%{x}<br>R_union=%{z:.2%}<extra></extra>",
		)
	)

	fig.update_layout(
		template="plotly_white",
		title="Group Topic Overlap Matrix (R_union)",
		xaxis_title="Group B",
		yaxis_title="Group A",
		width=980,
		height=880,
		margin={"l": 80, "r": 30, "t": 70, "b": 80},
	)

	fig.write_html(str(html_path), include_plotlyjs="cdn")


def format_panel_text(value, width=48):
	text = str(value).strip()
	if not text:
		return "-"
	wrapped_lines = []
	for paragraph in text.splitlines() or [text]:
		parts = textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False)
		wrapped_lines.extend(parts or [""])
	return "<br>".join(html.escape(line) for line in wrapped_lines)


def build_detail_table_values(point_info, neighbors, sims, top_k):
	if point_info is None:
		return [
			["Selection", "How to use", "Full text", f"Top {top_k} neighbors"],
			[
				"No amendment selected",
				"Click any point on the map to inspect its full content here.",
				"-",
				"-",
			],
		]

	neighbor_lines = []
	for rank, (neighbor, similarity) in enumerate(zip(neighbors, sims), start=1):
		neighbor_lines.append(
			f"#{rank} {html.escape(str(neighbor['number']))} ({html.escape(str(neighbor['author_group']))}) - cos={similarity:.3f}"
		)

	metadata = (
		f"Number: {html.escape(str(point_info['number']))}<br>"
		f"Group: {html.escape(str(point_info['author_group']))}<br>"
		f"Topic: {html.escape(str(point_info['topic']))}"
	)

	return [
		["Selection", "How to use", "Full text", f"Top {top_k} neighbors"],
		[
			metadata,
			"Click another point to move the selection and refresh the neighbor links.",
			format_panel_text(point_info["summary"], width=56),
			"<br>".join(neighbor_lines) if neighbor_lines else "-",
		],
	]


def build_interactive_map(df, coords, neighbors_idx, neighbors_sim, html_path, k, topics=None, topic_meta_df=None):
	x = coords[:, 0]
	y = coords[:, 1]

	palette = px.colors.qualitative.Safe + px.colors.qualitative.Bold + px.colors.qualitative.Set3
	groups = sorted(df["author_group"].dropna().astype(str).unique().tolist())
	color_map = {}
	unknown_idx = 0
	for group in groups:
		if group in COLORS:
			color_map[group] = COLORS[group]
		else:
			color_map[group] = palette[unknown_idx % len(palette)]
			unknown_idx += 1

	fig = make_subplots(
		rows=1,
		cols=2,
		column_widths=[0.72, 0.28],
		horizontal_spacing=0.04,
		specs=[[{"type": "scatter"}, {"type": "table"}]],
	)

	# Add topic hulls as semi-transparent backgrounds if topics provided.
	if topics is not None:
		topic_palette = px.colors.qualitative.Pastel + px.colors.qualitative.Light24
		topic_labels_x = []
		topic_labels_y = []
		topic_labels_text = []
		topic_labels_hover = []
		hulls = compute_topic_hulls(coords, topics)
		topic_meta = {}
		if topic_meta_df is not None and not topic_meta_df.empty:
			topic_meta = {int(row.topic): row for row in topic_meta_df.itertuples(index=False)}
		for topic, hull_x, hull_y in hulls:
			topic_color = topic_palette[abs(topic) % len(topic_palette)]
			meta = topic_meta.get(int(topic))
			hover_text = f"Topic {topic}"
			keywords = "N/A"
			legend_keywords = ""
			if meta is not None:
				keywords = meta.top_words
				legend_keywords = f" | {keywords}" if keywords and keywords != "N/A" else ""
				hover_text = (
					f"Topic {topic}"
					f"<br>Size: {meta.size}"
					f"<br>Share: {meta.share:.1%}"
					f"<br>Dominant group: {meta.dominant_group} ({meta.dominant_group_share:.1%})"
					f"<br>Keywords: {meta.top_words}"
				)
			mask = topics == topic
			pts = coords[mask]
			if len(pts):
				centroid = pts.mean(axis=0)
				topic_labels_x.append(float(centroid[0]))
				topic_labels_y.append(float(centroid[1]))
				topic_labels_text.append(f"Topic {topic}<br>{keywords}")
				topic_labels_hover.append(hover_text)
			fig.add_trace(
				go.Scatter(
					x=hull_x,
					y=hull_y,
					mode="lines",
					name=f"Topic {topic}{legend_keywords}",
					line={"width": 0, "color": topic_color},
					fill="toself",
					fillcolor=topic_color,
					opacity=0.15,
					text=[hover_text] * len(hull_x),
					hovertemplate="%{text}<extra></extra>",
					showlegend=True,
					meta={"topic": int(topic)},
				)
				,
				row=1,
				col=1,
			)

		if topic_labels_x:
			fig.add_trace(
				go.Scatter(
					x=topic_labels_x,
					y=topic_labels_y,
					mode="text",
					text=topic_labels_text,
					textposition="middle center",
					textfont={"size": 11, "color": "#111111"},
					hovertext=topic_labels_hover,
					hovertemplate="%{hovertext}<extra></extra>",
					showlegend=False,
					name="Topic keywords",
				)
				,
				row=1,
				col=1,
			)

	# Empty edge trace, filled dynamically on click.
	fig.add_trace(
		go.Scatter(
			x=[],
			y=[],
			mode="lines",
			line={"width": 1.5, "color": "#4a4a4a"},
			hoverinfo="skip",
			showlegend=False,
			name="Connections",
		),
		row=1,
		col=1,
	)
	edge_trace_idx = len(fig.data) - 1

	# Selected amendment marker.
	fig.add_trace(
		go.Scatter(
			x=[],
			y=[],
			mode="markers",
			marker={"size": 13, "symbol": "x", "color": "#111111"},
			hoverinfo="skip",
			showlegend=False,
			name="Selected",
		),
		row=1,
		col=1,
	)
	selected_trace_idx = len(fig.data) - 1

	df_plot = df.copy()
	df_plot["author_group"] = df_plot["author_group"].fillna("UNKNOWN").astype(str)

	for group, group_df in df_plot.groupby("author_group", sort=True):
		idxs = group_df.index.to_numpy()
		hover_text = []
		for idx in idxs:
			summary = str(df_plot.iloc[idx]["amendment_summary"])
			summary_short = summary[:260] + "..." if len(summary) > 260 else summary
			topic_str = f" | topic={df_plot.iloc[idx].get('topic', 'N/A')}" if "topic" in df_plot.columns else ""
			hover_text.append(
				f"number={df_plot.iloc[idx]['number']}<br>group={group}{topic_str}<br>{summary_short}"
			)

		fig.add_trace(
			go.Scattergl(
				x=x[idxs],
				y=y[idxs],
				mode="markers",
				name=group,
				customdata=idxs,
				text=hover_text,
				hovertemplate="%{text}<extra></extra>",
				marker={
					"size": 7,
					"opacity": 0.9,
					"color": color_map.get(group, "#7f7f7f"),
				},
			),
			row=1,
			col=1,
		)

	point_details = []
	for idx, row in df_plot.iterrows():
		point_details.append(
			{
				"idx": int(idx),
				"number": str(row["number"]),
				"author_group": str(row["author_group"]),
				"topic": "N/A" if "topic" not in df_plot.columns or pd.isna(row.get("topic")) else str(row.get("topic")),
				"summary": str(row["amendment_summary"]),
			}
		)

	default_labels, default_values = build_detail_table_values(None, [], [], k)
	fig.add_trace(
		go.Table(
			header={
				"values": ["Field", "Value"],
				"fill_color": "#0f172a",
				"font": {"color": "white", "size": 13},
				"align": "left",
				"height": 30,
			},
			cells={
				"values": [default_labels, default_values],
				"align": "left",
				"fill_color": [["#f8fafc", "#ffffff", "#f8fafc", "#ffffff"], ["#ffffff", "#ffffff", "#ffffff", "#ffffff"]],
				"font": {"color": "#0f172a", "size": 12},
				"height": 110,
			},
			columnwidth=[110, 420],
		),
		row=1,
		col=2,
	)
	detail_trace_idx = len(fig.data) - 1

	neighbors_json = json.dumps(neighbors_idx.tolist())
	sims_json = json.dumps(np.round(neighbors_sim, 6).tolist())
	xs_json = json.dumps(x.tolist())
	ys_json = json.dumps(y.tolist())
	point_details_json = json.dumps(point_details)

	post_script = "const gd = document.getElementById('{plot_id}');\n"
	post_script += f"const neighbors = {neighbors_json};\n"
	post_script += f"const sims = {sims_json};\n"
	post_script += f"const xs = {xs_json};\n"
	post_script += f"const ys = {ys_json};\n"
	post_script += f"const pointDetails = {point_details_json};\n"
	post_script += f"const edgeTraceIdx = {edge_trace_idx};\n"
	post_script += f"const selectedTraceIdx = {selected_trace_idx};\n"
	post_script += f"const detailTraceIdx = {detail_trace_idx};\n"
	post_script += """
function escapeHtml(value) {
	return String(value)
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;');
}

function wrapText(value, width) {
	const text = String(value || '').trim();
	if (!text) return '-';
	const words = text.split(/\s+/);
	const lines = [];
	let current = '';
	words.forEach(function(word) {
		const next = current ? `${current} ${word}` : word;
		if (next.length > width && current) {
			lines.push(current);
			current = word;
		} else {
			current = next;
		}
	});
	if (current) lines.push(current);
	return lines.map(escapeHtml).join('<br>');
}

function renderPointDetail(idx) {
	const info = pointDetails[idx];
	if (!info) return;
	const neighborLines = neighbors[idx].map(function(neighborIdx, rank) {
		const neighbor = pointDetails[neighborIdx];
		const similarity = sims[idx][rank];
		if (!neighbor) return `#${rank + 1} ${neighborIdx}`;
		return `#${rank + 1} ${escapeHtml(neighbor.number)} (${escapeHtml(neighbor.author_group)}) - cos=${similarity.toFixed(3)}`;
	});

	const labels = ['Selection', 'How to use', 'Full text', `Top ${neighbors[idx].length} neighbors`];
	const values = [
		`Number: ${escapeHtml(info.number)}<br>Group: ${escapeHtml(info.author_group)}<br>Topic: ${escapeHtml(info.topic)}`,
		'Click another point to move the selection and refresh the neighbor links.',
		wrapText(info.summary, 56),
		neighborLines.length ? neighborLines.join('<br>') : '-'
	];

	Plotly.restyle(gd, {'cells.values': [[labels, values]]}, [detailTraceIdx]);
}

gd.on('plotly_click', function(evt) {
  if (!evt || !evt.points || evt.points.length === 0) return;
  const point = evt.points[0];

  if (typeof point.customdata === 'undefined' || point.customdata === null) return;
  const idx = Number(point.customdata);
  const edgeX = [];
  const edgeY = [];
  for (let n = 0; n < neighbors[idx].length; n++) {
    const j = neighbors[idx][n];
    edgeX.push(xs[idx], xs[j], null);
    edgeY.push(ys[idx], ys[j], null);
  }
	Plotly.restyle(gd, {x: [edgeX], y: [edgeY]}, [edgeTraceIdx]);
	Plotly.restyle(gd, {x: [[xs[idx]]], y: [[ys[idx]]]}, [selectedTraceIdx]);
	renderPointDetail(idx);
});
"""

	fig.update_layout(
		template="plotly_white",
		title=f"PLFSS Map (2D PCA) - topic keywords shown on clusters, click a point to show its top {k} neighbors",
		legend={
			"title": {"text": "author_group / topic"},
			"orientation": "h",
			"x": 0.0,
			"xanchor": "left",
			"y": -0.12,
			"yanchor": "top",
			"traceorder": "grouped",
			"groupclick": "toggleitem",
			"bgcolor": "rgba(255,255,255,0.9)",
		},
		width=1500,
		height=900,
		margin={"l": 30, "r": 30, "t": 70, "b": 120},
	)

	fig.update_xaxes(title_text="Component 1", row=1, col=1)
	fig.update_yaxes(title_text="Component 2", row=1, col=1)

	fig.write_html(str(html_path), include_plotlyjs="cdn", post_script=post_script)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--csv",
		default="corpus/post-train/plfss_2025_finetuned_embed.csv",
		help="Input CSV with amendment_summary embeddings",
	)
	parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
	parser.add_argument(
		"--neighbors-out",
		default="corpus/post-train/plfss_2025_topk_neighbors.csv",
		help="Output CSV containing top-k neighbors for each amendment",
	)
	parser.add_argument(
		"--html-out",
		default="corpus/post-train/plfss_2025_map.html",
		help="Output HTML path for the interactive 2D map",
	)
	parser.add_argument(
		"--group-matrix-counts-out",
		default="corpus/post-train/plfss_2025_group_neighbor_counts.csv",
		help="Output CSV path for group-to-group neighbor counts",
	)
	parser.add_argument(
		"--group-matrix-share-out",
		default="corpus/post-train/plfss_2025_group_neighbor_share.csv",
		help="Output CSV path for row-normalized group-to-group neighbor shares",
	)
	parser.add_argument(
		"--group-matrix-html-out",
		default="corpus/post-train/plfss_2025_group_neighbor_matrix.html",
		help="Output HTML path for confusion-matrix style neighbor group heatmap",
	)
	parser.add_argument(
		"--topics-out",
		default="corpus/post-train/plfss_2025_topics.csv",
		help="Output CSV path for topic assignments and metadata",
	)
	parser.add_argument(
		"--topic-overview-html-out",
		default="corpus/post-train/plfss_2025_topic_overview.html",
		help="Output HTML path for topic-cluster overview",
	)
	parser.add_argument(
		"--min-topic-size",
		type=int,
		default=5,
		help="Minimum number of documents per topic for BERTopic",
	)
	args = parser.parse_args()

	if args.k <= 0:
		raise ValueError("k must be strictly positive")

	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(
		csv_path,
		converters={"embedding_amendment_summary": parse_embedding},
	)

	required_cols = {"number", "author_group", "amendment_summary", "embedding_amendment_summary"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

	df = df.dropna(subset=["amendment_summary", "embedding_amendment_summary"]).reset_index(drop=True)

	embeddings = np.array(df["embedding_amendment_summary"].tolist(), dtype=float)
	neighbors_idx, neighbors_sim = compute_top_k_neighbors(embeddings, args.k)

	neighbors_df = build_neighbors_dataframe(df, neighbors_idx, neighbors_sim)
	neighbors_out = Path(args.neighbors_out)
	neighbors_out.parent.mkdir(parents=True, exist_ok=True)
	neighbors_df.to_csv(neighbors_out, index=False)

	# Topic modeling with BERTopic
	model, topics = fit_topic_model(embeddings, df["amendment_summary"].tolist(), min_topic_size=args.min_topic_size)
	df["topic"] = topics
	topic_meta_df = build_topic_metadata(df, topics, model)

	topics_out = Path(args.topics_out)
	topics_out.parent.mkdir(parents=True, exist_ok=True)
	topic_export = df[["number", "author_group", "amendment_summary", "topic"]].copy()
	topic_export = topic_export.merge(topic_meta_df, on="topic", how="left")
	topic_export.to_csv(topics_out, index=False)
	print(f"Saved topic assignments: {topics_out}")

	topic_overview_html_out = Path(args.topic_overview_html_out)
	topic_overview_html_out.parent.mkdir(parents=True, exist_ok=True)
	save_topic_cluster_overview(topic_meta_df, topic_overview_html_out)

	# Group matrices from topic-cluster sharing (not neighbor retrieval).
	group_counts, group_share = build_group_topic_cluster_matrices(df)
	group_counts_out = Path(args.group_matrix_counts_out)
	group_share_out = Path(args.group_matrix_share_out)
	group_counts_out.parent.mkdir(parents=True, exist_ok=True)
	group_share_out.parent.mkdir(parents=True, exist_ok=True)
	group_counts.to_csv(group_counts_out)
	group_share.to_csv(group_share_out)

	group_matrix_html_out = Path(args.group_matrix_html_out)
	group_matrix_html_out.parent.mkdir(parents=True, exist_ok=True)
	save_group_neighbor_heatmap(group_share, group_matrix_html_out)

	# Compute PCA and build interactive map with topics
	coords = pca_2d(embeddings)
	html_out = Path(args.html_out)
	html_out.parent.mkdir(parents=True, exist_ok=True)
	build_interactive_map(
		df,
		coords,
		neighbors_idx,
		neighbors_sim,
		html_out,
		args.k,
		topics=topics,
		topic_meta_df=topic_meta_df,
	)

	print(f"Loaded PLFSS amendments: {len(df)}")
	print(f"Number of topics: {len(np.unique(topics[topics != -1]))}")
	print(f"Saved top-k neighbors CSV: {neighbors_out}")
	print(f"Saved interactive map HTML: {html_out}")
	print(f"Saved group-topic overlap intersection-count matrix CSV: {group_counts_out}")
	print(f"Saved group-topic overlap R_union matrix CSV: {group_share_out}")
	print(f"Saved group-topic overlap R_union heatmap HTML: {group_matrix_html_out}")
	print(f"Saved topic overview HTML: {topic_overview_html_out}")


if __name__ == "__main__":
	main()
