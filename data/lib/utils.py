from thefuzz.fuzz import ratio
from collections import Counter

import re

APOS = re.compile(r"(d|l|qu|n)['’]")
        
def fuzz_choice(occ, refs):
    refs = set(refs)
    cnts = Counter({ k: ratio(occ, k) for k in refs})
    return cnts.most_common(1)[0][0]

def fuzz_word(occ1, occ2):
    bits1 = set(APOS.sub('', occ1.lower()).split(' '))
    bits2 = set(APOS.sub('', occ2.lower()).split(' '))

    return len(bits1.intersection(bits2)) / min(len(bits1), len(bits2))


def fuzz_word_choice(occ, refs):
    refs = set(refs)
    cnts = Counter({ k: fuzz_word(occ, k) for k in refs})
    return cnts.most_common(1)[0][0]