import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_matrix(path):
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Matrix file not found: {p}")
	return pd.read_csv(p, index_col=0)


def align_matrices(pre, post):
	groups = sorted(set(pre.index) | set(pre.columns) | set(post.index) | set(post.columns))
	pre_aligned = pre.reindex(index=groups, columns=groups, fill_value=0.0)
	post_aligned = post.reindex(index=groups, columns=groups, fill_value=0.0)
	return pre_aligned, post_aligned


def compute_group_external_sharing_scores(share_matrix, counts_matrix=None):
	"""Average R_union overlap with other groups (excluding diagonal)."""
	scores = {}
	for group in share_matrix.index:
		row = share_matrix.loc[group].drop(index=group, errors="ignore")
		if row.empty:
			scores[group] = 0.0
			continue

		if counts_matrix is None:
			scores[group] = float(row.mean())
			continue

		weights = counts_matrix.loc[group].drop(index=group, errors="ignore").reindex(row.index, fill_value=0.0)
		total = float(weights.sum())
		if total == 0:
			scores[group] = float(row.mean())
		else:
			scores[group] = float((row * weights).sum() / total)

	return pd.Series(scores).reindex(share_matrix.index)


def compute_global_external_sharing(share_matrix, counts_matrix=None):
	"""Global average R_union overlap between different groups."""
	mask = np.ones_like(share_matrix.values, dtype=bool)
	np.fill_diagonal(mask, False)
	values = share_matrix.values[mask]
	if values.size == 0:
		return 0.0

	if counts_matrix is None:
		return float(values.mean())

	weights = counts_matrix.reindex(index=share_matrix.index, columns=share_matrix.columns, fill_value=0.0).values[mask]
	total = float(weights.sum())
	if total == 0:
		return float(values.mean())
	return float((values * weights).sum() / total)


def build_comparison_figure(pre_share, post_share, global_pre, global_post):
	delta_share = post_share - pre_share
	text_values = np.round(delta_share.values * 100, 1)

	fig = go.Figure(
		data=go.Heatmap(
			z=delta_share.values,
			x=delta_share.columns,
			y=delta_share.index,
			colorscale="RdBu",
			zmid=0,
			text=text_values,
			texttemplate="%{text:+.1f}%",
			textfont={"size": 11},
			colorbar={"title": "delta R_union"},
			hovertemplate=(
				"group_a=%{y}<br>group_b=%{x}<br>"
				+ "pre=%{customdata[0]:.2%}<br>post=%{customdata[1]:.2%}<br>delta=%{z:+.2%}<extra></extra>"
			),
			customdata=np.dstack((pre_share.values, post_share.values)),
		)
	)

	fig.update_layout(
		template="plotly_white",
		width=980,
		height=880,
		title=(
			"Evolution Matrix of Group Topic Overlap R_union (Post - Pre)"
			f"<br><sup>global pre={global_pre:.3f} | global post={global_post:.3f} | delta={global_post - global_pre:+.3f}</sup>"
		),
		margin={"l": 80, "r": 30, "t": 90, "b": 80},
	)

	fig.update_xaxes(title_text="Group B", tickangle=45)
	fig.update_yaxes(title_text="Group A")

	return fig


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pre-share",
		default="corpus/pre-train/amendements_group_neighbor_share.csv",
		help="Pre-train group topic-overlap R_union matrix CSV",
	)
	parser.add_argument(
		"--post-share",
		default="corpus/post-train/amendements_group_neighbor_share.csv",
		help="Post-train group topic-overlap R_union matrix CSV",
	)
	parser.add_argument(
		"--pre-counts",
		default="corpus/pre-train/amendements_group_neighbor_counts.csv",
		help="Optional pre-train pair-count matrix for weighted external sharing",
	)
	parser.add_argument(
		"--post-counts",
		default="corpus/post-train/amendements_group_neighbor_counts.csv",
		help="Optional post-train pair-count matrix for weighted external sharing",
	)
	parser.add_argument(
		"--out-csv",
		default="corpus/comparison/confusion_evolution_by_group.csv",
		help="Output CSV for external R_union evolution by group",
	)
	parser.add_argument(
		"--out-html",
		default="corpus/comparison/confusion_evolution.html",
		help="Output HTML with R_union evolution heatmap",
	)
	args = parser.parse_args()

	pre_share = load_matrix(args.pre_share)
	post_share = load_matrix(args.post_share)
	pre_share, post_share = align_matrices(pre_share, post_share)

	pre_counts = load_matrix(args.pre_counts) if Path(args.pre_counts).exists() else None
	post_counts = load_matrix(args.post_counts) if Path(args.post_counts).exists() else None
	if pre_counts is not None:
		pre_counts, _ = align_matrices(pre_counts, pre_counts)
	if post_counts is not None:
		post_counts, _ = align_matrices(post_counts, post_counts)

	pre_external = compute_group_external_sharing_scores(pre_share, pre_counts)
	post_external = compute_group_external_sharing_scores(post_share, post_counts)

	by_group = pd.DataFrame(
		{
			"pre_external_r_union": pre_external,
			"post_external_r_union": post_external,
			"delta_external_r_union": post_external - pre_external,
		}
	).sort_index()

	global_pre = compute_global_external_sharing(pre_share, pre_counts)
	global_post = compute_global_external_sharing(post_share, post_counts)

	out_csv = Path(args.out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	by_group.to_csv(out_csv)

	fig = build_comparison_figure(pre_share, post_share, global_pre, global_post)
	out_html = Path(args.out_html)
	out_html.parent.mkdir(parents=True, exist_ok=True)
	fig.write_html(str(out_html), include_plotlyjs="cdn")

	print(f"Saved external R_union evolution by group: {out_csv}")
	print(f"Saved R_union evolution figure: {out_html}")
	print(f"Global external R_union pre-train: {global_pre:.6f}")
	print(f"Global external R_union post-train: {global_post:.6f}")
	print(f"Global external R_union delta (post-pre): {global_post - global_pre:+.6f}")


if __name__ == "__main__":
	main()
