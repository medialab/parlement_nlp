import casanova
import argparse
import sys
import random
import itertools as it

random.seed(42)

def err(*args):
    print(*args, file=sys.stderr)

def make_paires(input_path, output_path):
    with casanova.reader(input_path) as reader:
        header = reader.fieldnames
        rows = list(reader)
        header = ['a_' + h for h in header] + ['b_' + h for h in header] + ["agreement"]
    
    with casanova.writer(output_path, header) as writer:
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

            item = a + b + [1 if a_vote == b_vote else 0]
            writer.writerow(item)

            j += 1
            if j == len(rows):
                j -= 1
                i += 1


def make_triplets(input_path, output_path):
    
    with casanova.reader(input_path) as reader:
        header = reader.fieldnames
        rows = list(reader)
        header = ['anc_' + h for h in header] + ['pos_' + h for h in header] + ['neg_' + h for h in header]

    rows = filter(lambda a: a[10] == "amendment", rows)
    groups = it.groupby(rows, lambda a: a[1])

    with casanova.writer(output_path, header) as writer:
        for _, group in groups:
            group = list(group)

            # On récupère les rows qui concernent des interventions 
            # en lien avec la défense d'un amendement
            precised = [it for it in group if it[13]]
            
            # On récupère – à l'inverse | les rows qui concernent
            # des interventions en lien avec la réponse à des amendements
            reactions = [item for item in group if not item[13]]

            positifs = [item for item in reactions if item[9] == "POUR"]
            negatifs = [item for item in reactions if item[9] == "CONTRE"]

            for a in precised:
                for b in positifs:
                    for c in negatifs:
                        item = a + b + c
                        writer.writerow(item)

           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Turn main dataset into paires and triplets datasets')
    parser.add_argument('input', help='Path of main dataset')
    parser.add_argument('output_paires', help="Output path for paires")
    parser.add_argument('output_triplets', help="Output path for triplets")

    args = parser.parse_args()

    make_paires(args.input, args.output_paires)
    make_triplets(args.input, args.output_triplets)