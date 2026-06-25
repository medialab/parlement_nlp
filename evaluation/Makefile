sick:
	cat datasets/test_sick.csv | plonge 'sentence_A,sentence_B' -m 'Qwen/Qwen3-Embedding-0.6B' -d cuda -b 1 -p 4906 > embeddings/pre-train/test_sick.csv
	cat datasets/test_sick.csv | plonge 'sentence_A,sentence_B' -m '../training/checkpoints/trial-16/checkpoint-4214' -d cuda -b 1 -p 4906 > embeddings/post-train/test_sick.csv

sts:
	cat datasets/test_sts.csv | plonge 'sentence1,sentence2' -m 'Qwen/Qwen3-Embedding-0.6B' -d cuda -b 1 -p 29973 > embeddings/pre-train/test_sts.csv
	cat datasets/test_sts.csv | plonge 'sentence1,sentence2' -m '../training/checkpoints/trial-16/checkpoint-4214' -d cuda -b 1 -p 29973 > embeddings/post-train/test_sts.csv

spearman:
	cat datasets/test_spearman.csv | plonge 'a_speech,b_speech' -m 'Qwen/Qwen3-Embedding-0.6B' -d cuda -b 1 -p 1000 > embeddings/pre-train/test_spearman.csv
	cat datasets/test_spearman.csv | plonge 'a_speech,b_speech' -m '../training/checkpoints/trial-16/checkpoint-4214' -d cuda -b 1 -p 1000 > embeddings/post-train/test_spearman.csv