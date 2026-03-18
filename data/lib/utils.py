from thefuzz.fuzz import ratio
from collections import Counter
        
def fuzz_choice(occ, refs):
    refs = set(refs)
    cnts = Counter({ k: ratio(occ, k) for k in refs})
    return cnts.most_common(1)[0][0]