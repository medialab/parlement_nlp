import casanova
import argparse
import sys
import random

random.seed(42)

parser = argparse.ArgumentParser(description='Process CSV data in parallel')
parser.add_argument('input', help='Input CSV file')
args = parser.parse_args()

def err(*args):
    print(*args, file=sys.stderr)

with casanova.reader(args.input) as reader:
    header = reader.fieldnames
    rows = list(reader)
    size = len(rows)
    header = ['a_' + h for h in header] + ['b_' + h for h in header] + ["agreement"]

#groups = [(a, list(b)) for a, b in it.groupby(rows, lambda a: a[1])]

with casanova.writer(sys.stdout, header) as writer:
    i, j = 0, 1
    while i < (len(rows) - 1):        
        a, b = rows[i], rows[j]
        
        a_source = a[10]
        b_source = b[10]

        if a_source == "amendment" or b_source == "amendment":
            i += 1
            j = i + 1
            continue

        a_debate_id = a[2]
        b_debate_id = b[2]

        a_vote = a[9]
        b_vote = b[9]

        if a_debate_id != b_debate_id:
            i += 1
            j = i + 1
            continue

        """
        if random.choice(range(3)) == 0:
            while True:
                ci = random.randrange(size)
                c = rows[ci]
                if c[2] != a_debate_id:
                    item = a + c + [-1]
                    writer.writerow(item)
                    break
        else:
            item = a + b + [1 if a_vote == b_vote else 0]

            writer.writerow(item)

            j += 1

            if j == len(rows):
                j -= 1
                i += 1
        """

        item = a + b + [1 if a_vote == b_vote else 0]
        writer.writerow(item)

        j += 1
        if j == len(rows):
            j -= 1
            i += 1

# uv run parallele.py ./export/dataset-v3.csv > ./parallele4.csv
# xan search "VTANR5L15V3060-15-248481" -v < parallele4.csv > parallele5.csv