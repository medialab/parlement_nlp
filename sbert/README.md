# Sentence BERT (des)agreement

training script to fine-tune SentenceBERT model according to agreement/desagreement pairs of speeches from the french parlement.

Put `parallele.csv` in `./dataset`. THis file is a set of pairs of successive speeches spoken in the Assemblée Nationale. Only successive speeches of the same debate are kept (same debate = debate on the same amendment/article/motion de rejet/explication de vote).

File `train.py` is a training script to fine-tune a SentenceBERT model (based on ModernCamembert-base) by the confrontation of two pairs, a distance signal (-1 or 1) and a [CoSENTLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss).

