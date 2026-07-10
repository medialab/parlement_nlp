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

### Qwen post-train (LoRA 16 alpha 32)

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

### Qwen post-train (LoRA 8 alpha 8)

```
==== PARLEMENT TEST SET - KL DIV ====
KL-Divergence : 2.6058231048618286
Plot of distances distribution saved to ./figures/parlement_lora8.jpg

==== Catie-AQ/STS ====
[musts_french] pearson correlation : 0.8421486929785085 ; spearman correlation : 0.8634071986071521
[opusparcus] pearson correlation : 0.6206446616916905 ; spearman correlation : 0.5759774040447815
[ordalie] pearson correlation : 0.705687862318254 ; spearman correlation : 0.7040332711471524
[sick] pearson correlation : 0.7768973025313913 ; spearman correlation : 0.7400230913207326
[sts12] pearson correlation : 0.7117014155582155 ; spearman correlation : 0.6512918089371738
[sts13] pearson correlation : 0.6994222289485011 ; spearman correlation : 0.7163492416965946
[sts14] pearson correlation : 0.6607606323543662 ; spearman correlation : 0.6738077384343477
[sts15] pearson correlation : 0.7705682405713421 ; spearman correlation : 0.7826996371466706
[sts16] pearson correlation : 0.6886070930291002 ; spearman correlation : 0.7293389356079929
[sts22] pearson correlation : 0.7401977618980793 ; spearman correlation : 0.7654268243418728
[stsb] pearson correlation : 0.7675855360911429 ; spearman correlation : 0.771163263035406

==== PARLEMENT TEST SET - SPEARMAN ====
Pearson correlation : 0.12150161331693211 ; spearman correlation : 0.14692647936771078
```

### Qwen post-train (LoRA 4 alpha 4)

```
==== PARLEMENT TEST SET - KL DIV ====
KL-Divergence : 1.177160658778887
Plot of distances distribution saved to ./figures/parlement_lora4.jpg

==== Catie-AQ/STS ====
[musts_french] pearson correlation : 0.8531249761225186 ; spearman correlation : 0.8724872808595604
[opusparcus] pearson correlation : 0.6259193904634187 ; spearman correlation : 0.5803498427907282
[ordalie] pearson correlation : 0.731433255566956 ; spearman correlation : 0.7224462065553319
[sick] pearson correlation : 0.7916901142006121 ; spearman correlation : 0.7504733261932526
[sts12] pearson correlation : 0.7239128303844719 ; spearman correlation : 0.6592916007415014
[sts13] pearson correlation : 0.7167824630342192 ; spearman correlation : 0.7308118852557652
[sts14] pearson correlation : 0.6828973132987202 ; spearman correlation : 0.6940627041299257
[sts15] pearson correlation : 0.7927470147459597 ; spearman correlation : 0.8026432211137132
[sts16] pearson correlation : 0.7136780446498019 ; spearman correlation : 0.747165329484008
[sts22] pearson correlation : 0.7754487402995835 ; spearman correlation : 0.7943898483643012
[stsb] pearson correlation : 0.7901524003005924 ; spearman correlation : 0.7933174933900895

==== SICK TEST SET ====
Plot of distances distribution saved to ./figures/sick_lora4.jpg

==== PARLEMENT TEST SET - SPEARMAN ====
Pearson correlation : 0.21520847167892565 ; spearman correlation : 0.23386854517548195
```

### Qwen post-train (LoRA 2 alpha 2)

```
==== PARLEMENT TEST SET - KL DIV ====
KL-Divergence : 0.40694567057643244
Plot of distances distribution saved to ./figures/parlement_lora2.jpg

==== Catie-AQ/STS ====
[musts_french] pearson correlation : 0.8702203804234885 ; spearman correlation : 0.886266862911588
[opusparcus] pearson correlation : 0.629851215431783 ; spearman correlation : 0.5809156106653454
[ordalie] pearson correlation : 0.7531797398970588 ; spearman correlation : 0.7309300463152678
[sick] pearson correlation : 0.8016798625476922 ; spearman correlation : 0.7597570440597363
[sts12] pearson correlation : 0.738319187602006 ; spearman correlation : 0.6709589883421188
[sts13] pearson correlation : 0.7268145337483329 ; spearman correlation : 0.7381245937162877
[sts14] pearson correlation : 0.6950201391488169 ; spearman correlation : 0.7025121297974665
[sts15] pearson correlation : 0.8070229456368313 ; spearman correlation : 0.8147889274174199
[sts16] pearson correlation : 0.7281042613103293 ; spearman correlation : 0.7571225738482003
[sts22] pearson correlation : 0.7941261376658973 ; spearman correlation : 0.8073373722963161
[stsb] pearson correlation : 0.7962988066930894 ; spearman correlation : 0.7968673696252412

==== SICK TEST SET ====
Plot of distances distribution saved to ./figures/sick_lora2.jpg

==== PARLEMENT TEST SET - SPEARMAN ====
Pearson correlation : 0.349726656960742 ; spearman correlation : 0.3574538961898274
```