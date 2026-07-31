import pandas as pd

if __name__ == "__main__":
    dfa = pd.read_csv("./datasets/test_kl.csv")
    dfb = pd.read_csv("./embeddings/pre-train/test_parlement_spearman.csv")

    dfa_a = dfa["a_speech"].tolist()
    dfa_b = dfa["b_speech"].tolist()
    dfa_s = dfa["score"].tolist()

    mapping = { (a,b) : s for (a, b, s) in zip(dfa_a, dfa_b, dfa_s)}

    def transform(a, b, score):
        if (a, b) in mapping:
            s = mapping[(a, b)]
            s = 0.5 if s == 0.0 else 1.0
            return s
        else:
            return score


    dfb["score"] = dfb.apply(lambda x: transform(x.a_speech, x.b_speech, x.score), axis=1)

    dfb.to_csv("./embeddings/pre-train/test_parlement_spearman.csv", index=None)
