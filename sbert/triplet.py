import casanova
import argparse
import sys
import random
import itertools as it

random.seed(42)

parser = argparse.ArgumentParser(description='Process CSV data in parallel')
parser.add_argument('input', help='Input CSV file')
args = parser.parse_args()

def err(*args):
    print(*args, file=sys.stderr)

def successives(iter):
    for a, b in it.permutations(iter, 2):
        a_vote = a[9]
        b_vote = b[9]

        if not (a_vote == "POUR" and b_vote == "CONTRE"):
            continue

        yield a, b

with casanova.reader(args.input) as reader:
    header = reader.fieldnames
    rows = list(reader)
    size = len(rows)
    header = ['anc_' + h for h in header] + ['pos_' + h for h in header] + ['neg_' + h for h in header]

rows = filter(lambda a: a[10] == "amendment", rows)
groups = it.groupby(rows, lambda a: a[1])

with casanova.writer(sys.stdout, header) as writer:
    for _, group in groups:
        group = list(group)

        precised = [it for it in group if it[13]]
        
        reactions = [item for item in group if not item[13]]
        reactions = successives(reactions)

        for a, (b, c) in it.product(precised, reactions):
            a_vote = a[9]
            b_vote = b[9]
            c_vote = c[9]

            item = a + b + c
            writer.writerow(item)


# uv run parallele.py ./export/dataset-v3.csv > ./parallele4.csv
# xan search "VTANR5L15V3060-15-248481" -v < parallele4.csv > parallele5.csv