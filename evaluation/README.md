# Evaluation

## Corpus

### Test parlement

**kl**

Test split sur notre corpus de débats parlementaires.

**spearman**

Réunie des paires de textes issus de débats différents (score 0.0) et des textes de même débats (score 1.0, échantillon de `test_kl.csv`).

Commande xan :

```bash
$ xan cat rows <(xan select 'a_speech,b_speech,score' datasets/test_kl.csv| xan sample 500 --seed 42 | xan transform score '1.0') datasets/inter_sample.csv | xan shuffle --seed 42 | xan view > datasets/test_spearman.csv
```

## Results

### Qwen pre-train

```
==== PARLEMENT TEST SET ====
KL-Divergence : 0.14425689691806104
Plot of distances distribution saved to ./figures/parlement_qwen.jpg

==== PARLEMENT TEST SET - SPEARMAN ====
Pearson correlation : 0.6567117436987768 ; spearman correlation : 0.6526162859904867

==== Catie-HQ/STS ====
[musts_french] pearson correlation : 0.8664264866643996 ; spearman correlation : 0.8773640150535658
[opusparcus] pearson correlation : 0.6285672726839958 ; spearman correlation : 0.5832650245883015
[ordalie] pearson correlation : 0.8704011220825929 ; spearman correlation : 0.7906013496229152
[sick] pearson correlation : 0.8192840131834456 ; spearman correlation : 0.7652036304805973
[sts12] pearson correlation : 0.7263268294269045 ; spearman correlation : 0.6498429142014877
[sts13] pearson correlation : 0.7689315819769644 ; spearman correlation : 0.7741950193801576
[sts14] pearson correlation : 0.7370156988925354 ; spearman correlation : 0.7224954097134025
[sts15] pearson correlation : 0.8287309508164773 ; spearman correlation : 0.8328369051450639
[sts16] pearson correlation : 0.7601617307484891 ; spearman correlation : 0.7756863832231059
[sts22] pearson correlation : 0.8452791608724106 ; spearman correlation : 0.8375076504749375
[stsb] pearson correlation : 0.8119985320797176 ; spearman correlation : 0.8035815955247501
```

### Qwen post-train

```
==== PARLEMENT TEST SET ====
KL-Divergence : 1.5246605608622403
Plot of distances distribution saved to ./figures/parlement_finetuned.jpg

==== PARLEMENT TEST SET - SPEARMAN ====
Pearson correlation : 0.1230091178625455 ; spearman correlation : 0.14272105790422226

==== Catie-AQ/STS ====
[musts_french] pearson correlation : 0.8466024050597206 ; spearman correlation : 0.8669843881807671
[opusparcus] pearson correlation : 0.6199876241412696 ; spearman correlation : 0.5775408962070319
[ordalie] pearson correlation : 0.7321617916223275 ; spearman correlation : 0.7208613861602694
[sick] pearson correlation : 0.7747218734037368 ; spearman correlation : 0.7386857544194174
[sts12] pearson correlation : 0.7197838382293642 ; spearman correlation : 0.6635716147564473
[sts13] pearson correlation : 0.6990171399413438 ; spearman correlation : 0.7201482042777935
[sts14] pearson correlation : 0.6672263318452959 ; spearman correlation : 0.6807617291279049
[sts15] pearson correlation : 0.7723339563027205 ; spearman correlation : 0.7838668440256225
[sts16] pearson correlation : 0.6870590737987518 ; spearman correlation : 0.7265847218998991
[sts22] pearson correlation : 0.7332860539360029 ; spearman correlation : 0.7490347495605365
[stsb] pearson correlation : 0.7737727598259007 ; spearman correlation : 0.7760043132353488
```