amendements:
	cat corpus/amendements.csv | plonge 'amendment_summary' -m "Qwen/Qwen3-Embedding-0.6B" -b 1 -d cuda -p 13405 > corpus/amendements_qwen_embed.csv
	#cat corpus/amendements.csv | plonge 'amendment_summary' -m "../training/checkpoints/2026-07-02_14-27-08/trial-62/checkpoint-4214/" -b 1 -d cuda -p 13405 > corpus/amendements_finetuned_embed.csv
