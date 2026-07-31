import pandas as pd
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

qwen = pd.read_csv(
    "./data/embed_qwen_salles_shoot.csv",
    converters={"embedding_amendment_summary": literal_eval},
)
finetuned = pd.read_csv(
    "./data/embed_finetuned_salles_shoot.csv",
    converters={"embedding_amendment_summary": literal_eval},
)
size = len(qwen)

qwen_sim = cosine_similarity(qwen["embedding_amendment_summary"].to_list())
finetuned_sim = cosine_similarity(finetuned["embedding_amendment_summary"].to_list())

confusion_qwen = np.zeros((2, 2))
confusion_finetuned = np.zeros((2, 2))

pros = qwen[qwen["score"] == 1.0].index.tolist()
cons = qwen[qwen["score"] == 0.0].index.tolist()

pros_s = len(pros)
cons_s = len(cons)

print(qwen_sim)

# Qwen

for i, rx in enumerate(qwen.index):
    scores = Counter( {ix: si.item() for ix, si in zip(qwen.index.tolist(), qwen_sim[i])} )
    scores = []
    print("[score=%f; group=%s]" % (qwen.loc[rx]["score"], qwen.loc[rx]["author_group"]))
    for ix, si in scores.most_common():
        row = qwen.loc[ix]
        print("\t[score=%f; group=%s]" % (row["score"], row["author_group"]))
    print("")