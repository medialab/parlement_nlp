# Evaluation

## Results

### Qwen pre-train

```
==== PARLEMENT TEST SET ====
KL-Divergence : 0.14425689691806104
Plot of distances distribution saved to ./figures/parlement_qwen.jpg

==== Catie-HQ/STS ====
Pearson correlation : 0.814599560613478
Spearman correlation : 0.7967733818326944
```

### Qwen post-train

```
==== PARLEMENT TEST SET ====
KL-Divergence : 1.5246605608622403
Plot of distances distribution saved to ./figures/parlement_finetuned.jpg

==== Catie-HQ/STS ====
Pearson correlation : 0.7005894274634787
Spearman correlation : 0.6701638905998752
```