datasets:
	uv run duplet.py ./export/dataset-v3.csv > ./dataset/duplet.csv
	uv run triplet.py ./export/dataset-v3.csv > ./dataset/triplet.csv

train:
	uv run train_triplet.py
	uv run train_duplet.py
