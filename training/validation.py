import argparse
import pandas as pd
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from plotly.colors import qualitative


def _compute_axis_range(logs, metric_col, padding_ratio=0.05):
	values = pd.to_numeric(logs[metric_col], errors="coerce").dropna()
	if values.empty:
		return [0.0, 1.0]

	vmin = values.min()
	vmax = values.max()

	if vmin == vmax:
		base_padding = max(abs(vmin) * padding_ratio, 1e-6)
		return [vmin - base_padding, vmax + base_padding]

	padding = (vmax - vmin) * padding_ratio
	return [vmin - padding, vmax + padding]

def _build_trial_color_map(logs):
	trials = sorted(logs["trial"].dropna().unique())
	palette = qualitative.Plotly
	return {trial: palette[idx % len(palette)] for idx, trial in enumerate(trials)}


def _add_metric_traces(fig, logs, metric_col, row, show_legend, trial_color_map):
	sorted_logs = logs.sort_values(["trial", "epoch"]).copy()

	for trial in sorted_logs["trial"].dropna().unique():
		trial_logs = sorted_logs[sorted_logs["trial"] == trial]
		trial_name = f"trial {int(trial)}"
		trial_color = trial_color_map.get(trial)
		fig.add_trace(
			go.Scatter(
				x=trial_logs["epoch"],
				y=trial_logs[metric_col],
				mode="lines",
				line={"color": trial_color},
				name=trial_name,
				legendgroup=trial_name,
				showlegend=show_legend,
				hovertemplate=(
					"trial=%{fullData.name}<br>"
					"epoch=%{x}<br>"
					f"{metric_col}=%{{y:.4f}}"
					"<extra></extra>"
				),
			),
			row=row,
			col=1,
		)


def to_figures(logs):
	kl_range = _compute_axis_range(logs, "kl_divergence")
	spearman_range = _compute_axis_range(logs, "spearman_cosine")
	loss_range = _compute_axis_range(logs, "loss")
	trial_color_map = _build_trial_color_map(logs)
	
	fig = make_subplots(
		rows=2,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.08,
		#subplot_titles=("KL Divergence by Trial", "Spearman Cosine by Trial", "Loss Progression"),
		subplot_titles=("KL Divergence by Trial", "Spearman Cosine by Trial"),
	)

	_add_metric_traces(fig, logs, "kl_divergence", row=1, show_legend=True, trial_color_map=trial_color_map)
	_add_metric_traces(fig, logs, "spearman_cosine", row=2, show_legend=False, trial_color_map=trial_color_map)
	#_add_metric_traces(fig, logs, "loss", row=3, show_legend=False, trial_color_map=trial_color_map)

	fig.update_layout(
		title="Trials Metrics Evolution",
		template="plotly_white",
		legend_title="Trials",
		legend={"groupclick": "togglegroup"},
		height=1200,
	)
	fig.update_xaxes(title_text="Epoch", row=3, col=1)
	fig.update_yaxes(title_text="kl_divergence", range=kl_range, autorange=False, row=1, col=1)
	fig.update_yaxes(title_text="spearman_cosine", range=spearman_range, autorange=False, row=2, col=1)
	#fig.update_yaxes(title_text="loss", range=loss_range, autorange=False, row=3, col=1)
	


	return fig

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("logs", type=str)
	args = parser.parse_args()

	df = pd.read_csv(args.logs)
	fig = to_figures(df)
	fig.show()
