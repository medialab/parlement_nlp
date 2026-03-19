# 1. sélectionner les portions de débat pertinent (date, titre)
# 2. lancer la recherche de segments
import casanova
import tqdm
import re

from thefuzz.fuzz import ratio
from collections import Counter

from lib.utils import fuzz_choice
from lib.votes import Votes
import argparse

parser = argparse.ArgumentParser(description='Parse debate segments from CSV files')
parser.add_argument('--votes-json', help='Path to scrutins JSON file')
parser.add_argument('--debates-csv', help='Path to seances CSV file')
parser.add_argument('--votes-selection-csv', help='Path to scrutins selection CSV file')
parser.add_argument('--output-csv', help='Path to output CSV file')

args = parser.parse_args()

VOTES_JSON = args.votes_json
DEBATES_CSV = args.debates_csv
VOTES_SELECTION_CSV = args.votes_selection_csv
OUTPUT_CSV = args.output_csv

PRESIDENT = re.compile(r"^pr[ée]sidente?$", re.I)
PARA = re.compile(r'<\/p>\s*?<p>')
AMENDEMENT = re.compile(r'(amendements?[,\s]+(identiques?)?[,\s]*)((n[°os\s]*|\d+\s*|,\s*|à\s*|et\s*|rectifié\s*)+)')

MAJORITY_GROUPS = ("LAREM", "RE", "EPR")

GROUPS_GOVERNMENT = {
    # Ministre avec le mauvais "dernier groupe"
    "Bruno Le Maire": ("LAREM", "RE","EPR",),
    "Catherine Vautrin": ("LAREM", "RE","EPR",),
    "Manuel Valls": ("LAREM", "RE","EPR",),
    "Amélie de Montchalin": ("LAREM", "RE","EPR",),
    "Laurent Saint-Martin": ("LAREM", "RE","EPR",),

    # Ministre sans historique de mandat à l'AN
    "Christophe Béchu": ("HOR",),
    "Charlotte Caubel": ("HOR",),
    "Philippe Tabarot": ("DR",),
    "Rachida Dati": ("DR",),
    "Agnès Canayer": ("DR",),
    "Bruno Retailleau": ("DR",),
    "Laurence Garnier": ("DR",),
}

votes = Votes(VOTES_JSON)

with casanova.reader(DEBATES_CSV) as csv_debates:
    legis_header = csv_debates.fieldnames
    legis = list(csv_debates)

selection = list(casanova.reader(VOTES_SELECTION_CSV))

def is_rappel_reglement(interv):
    if interv == "Rappel au règlement" or interv == "Rappels au règlement":
        return True
    return False


def is_appel_amendement(interv, amendment = None):
    if "soutenir" not in interv:
        return False
    
    if "parole est à" not in interv:
        if "vous avez la parole" not in interv:
            return False

    if "amendement" not in interv:
        return False
    
    if amendment:
        if not re.search(fr'\b{amendment}\b', interv):
            return False
    
    return True

def get_start_debate(row, date, amendment):
    interventions = row[13]
    function = row[11]
    debate_date = row[3]

    if date != debate_date:
        return None
        
    interventions = interventions.lower()
    interventions = PARA.split(interventions.strip().strip('<p>').strip('</p>'))

    # On skip si ce n'est pas une intervention de présidence
    if type(function) == str and not PRESIDENT.match(function): return None

    for intervention in interventions:
        if is_appel_amendement(intervention, amendment):
            return intervention
        
    return None


def is_amendement_adoption(interv, amendment = None):
    if f"est adopté" in interv or f"est pas adopté" in interv:
        if amendment and re.search(fr'\b{amendment}\b', interv):
            return True
        elif not amendment and "amendement" in interv and "no " in interv:
            return True
    
    return False

def get_end_debate(row, date, amendment):
    interventions = row[13]
    function = row[11]
    debate_date = row[3]
    code = row[12]

    if date != debate_date:
        return None


    interventions = interventions.lower()
    interventions = PARA.split(interventions.strip().strip('<p>').strip('</p>'))

    # On skip si ce n'est pas une intervention de présidence
    if not code in ("DIDASCALIE", ""):
        if not PRESIDENT.match(function):
            return None

    for intervention in interventions:
        if is_amendement_adoption(intervention, amendment):
            return intervention
        
    return None

def get_group(row, scrutin_id):
    name, group, last_group, _ = row[7], row[9], row[10], row[11]

    if not group:
        vote = votes.from_group(scrutin_id, last_group)
        if vote: return last_group
        else:
            for g in GROUPS_GOVERNMENT.get(name, MAJORITY_GROUPS):
                vote = votes.from_group(scrutin_id, g)
                if vote: return g
            for g in MAJORITY_GROUPS:
                vote = votes.from_group(scrutin_id, g)
                if vote: return g
    
    return group

def get_vote(row, scrutin_id):
    name, group, last_group, function = row[7], row[9], row[10], row[11]

    if function == "président": return None
    vote = votes.from_name(scrutin_id, name)

    if not vote and not group:
        vote = votes.from_group(scrutin_id, last_group)
        if not vote:
            for g in GROUPS_GOVERNMENT.get(name, MAJORITY_GROUPS):
                vote = votes.from_group(scrutin_id, g)
                if vote: break
        if not vote:
            for g in MAJORITY_GROUPS:
                vote = votes.from_group(scrutin_id, g)
                if vote: break
    
    return vote


def get_segment(rows, meta_amendment):
    date, _, _, vote_id, amendement, _, _, _, _ = meta_amendment

    results = []
    has_drawer = False
    has_rappel_reglement = False
    for i in range(len(rows)):
        # On regarde si la ligne en cours est la fin d'un débat
        # (aka fin de discussion d'un amendement)
        start_row = rows[i]

        result_start = get_start_debate(start_row, date, amendement)

        if result_start:
            vote = get_vote(start_row, vote_id)
            start_row[9] = get_group(start_row, vote_id)
            results.append([vote] + start_row)

            j = i + 1
            while j < len(rows):
                roll_row = rows[j]
                result_end = get_end_debate(roll_row, date, amendement)

                if is_appel_amendement(roll_row[13].lower()):
                    has_drawer = True

                if is_rappel_reglement(roll_row[13]):
                    has_rappel_reglement = True

                vote = get_vote(roll_row, vote_id)
                roll_row[9] = get_group(roll_row, vote_id)

                results.append([vote] + roll_row)
                
                if result_end:
                    return has_drawer, has_rappel_reglement, results, "ok"
                else:
                    if is_amendement_adoption(roll_row[13].lower()):
                        has_drawer = True
                    j += 1
    
    return False, False, [], "no end intervention found" if results else "not start intervention found"

def head_section(section):
    parts = [s.strip() for s in section.split('>')]
    return parts[0]

def reduce_legis(meta):
    date, dossier, _, _, _, _, _, _, _ = meta
    
    sub = []
    sections = []
    for row in legis:
        if date != row[3]: continue
        
        sections.append(row[5])

        sub.append(row)
    
    if not sections:
        return [], "no corresponding date"

    good_section = fuzz_choice(dossier, sections)

    filtered = [row for row in sub if row[5] == good_section]

    return (filtered, "ok") if filtered else (filtered, "no segment related to vote")


with casanova.writer(OUTPUT_CSV, [
    "vote_id",
    "amendment",
    "author_name",
    "author_group",
    "amendment_content",
    "amendment_summary",
    "drawer",
    "rappel_reglement",
    "found",
    "log",
    "vote"
] + legis_header) as writer:
    for meta_amendment in tqdm.tqdm(selection, total=len(selection)):
        date, _, _, vote_id, amendment, author_name, author_group, amendment_content, amendment_summary = meta_amendment
        sub, log = reduce_legis(meta_amendment)
        
        if not sub:
            writer.writerow(
                [vote_id, amendment, author_name, author_group, amendment_content, amendment_summary, None, False, False, log, None] + [""] * 14
            )
            continue

        drawer, rappel, segment, log = get_segment(sub, meta_amendment)

        if not segment:
            writer.writerow(
                [vote_id, amendment, author_name, author_group, amendment_content, amendment_summary, drawer, rappel, False, log, None] + [""] * 14
            )

        for row in segment:
            writer.writerow(
                [vote_id, amendment, author_name, author_group, amendment_content, amendment_summary, drawer, rappel, True, log] + row
            )

