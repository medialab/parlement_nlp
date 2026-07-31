import casanova
import json
import regex
import ebbe
import sys

from thefuzz.fuzz import ratio
from collections import defaultdict, Counter

from lib.votes import Votes
import argparse

parser = argparse.ArgumentParser(
    description="Match scrutin data with amendments and votes"
)
parser.add_argument("--input", help="Input CSV file")
parser.add_argument("--votes", help="Votes JSON file")
#parser.add_argument("--amendments", help="Amendments CSV file")

args = parser.parse_args()

INPUT_CSV = args.input
VOTES_JSON = args.votes
#AMENDMENTS_CSV = args.amendments


AMENDEMENT = regex.compile(
    r"(?P<parent>(à l'|aux ))?amendements?( identiques?)?(?P<suivant> suivants?)?( de suppression)?( n° ?(?P<number>([A-Z]{2})?[0-9]+))?",
    flags=regex.UNICODE,
)

MOTION_REJET = regex.compile(
    r"motion de rejet",
    flags=regex.UNICODE | regex.I
)

BISTER = '(' + \
  '(?:un|duo|ter|quater|quin|sex?|sept|octo|novo|unde?|duode)?' + \
  '(?:dec|v[ie]c|tr[ie]c|quadrag|quinquag|sexag|septuag|octog|nonag)' + \
  'ies|semel|bis|ter|quater|' + \
  '(?:quinqu|sex|sept|oct|no[nv])ies' + \
  ')'

ARTICLE_VOTE = regex.compile(
    r"^l'[Aa]rticle\s+(?P<num>\w+)(\s+er)?(\s+(?P<sup>%s))?(\s+(?P<letter>[A-Z]+)?)" % BISTER,
    flags=regex.UNICODE
)

ARTICLE = regex.compile(
    r"(?P<pos>[Aa]près|[Aa]vant)?(\s|l.)*[Aa]rticle\s+(?P<num>\w+)(\s+er)?(\s+(?P<sup>%s))?(\s+(?P<letter>[A-Z]+)?)" % BISTER,
    flags=regex.UNICODE
)

ENSEMBLE = regex.compile(
    r"^l.ensemble",
    flags=regex.UNICODE
)

RAPPEL_REGLEMENT = regex.compile(
    r"appels? aux? r[èe]glement",
    flags=regex.UNICODE | regex.I
)

EXPLICATION_VOTE = regex.compile(
    r'explications? de votes?',
    flags=regex.UNICODE | regex.I
)

MAJORITY_GROUPS = ("LAREM", "RE", "EPR")

GROUPS_GOVERNMENT = {
    # Ministre avec le mauvais "dernier groupe"
    "Bruno Le Maire": (
        "LAREM",
        "RE",
        "EPR",
    ),
    "Catherine Vautrin": (
        "LAREM",
        "RE",
        "EPR",
    ),
    "Manuel Valls": (
        "LAREM",
        "RE",
        "EPR",
    ),
    "Amélie de Montchalin": (
        "LAREM",
        "RE",
        "EPR",
    ),
    "Laurent Saint-Martin": (
        "LAREM",
        "RE",
        "EPR",
    ),
    # Ministre sans historique de mandat à l'AN
    "Christophe Béchu": ("HOR",),
    "Charlotte Caubel": ("HOR",),
    "Philippe Tabarot": ("DR",),
    "Rachida Dati": ("DR",),
    "Agnès Canayer": ("DR",),
    "Bruno Retailleau": ("DR",),
    "Laurence Garnier": ("DR",),
}

HARMONIZED_GROUPS = {
    "FI": "LFI",
    "LAREM": "LREM",
    "SOC": "SOC",
    "UDI-AGIR": "UDI",
    "GDR": "GDR",
    "MODEM": "MODEM",
    "NG": "SOC",
    "LR": "LR",
    "LT": "LIOT",
    "AGIR-E": "AGIR-E",  # ??? dissidents LREM et UDI
    "UDI_I": "UDI",
    "NI": "NI",
    "UDI-A-I": "UDI",
    "LC": "UDI",
    "DEM": "MODEM",
    "EDS": "EDS",  # ??? dissidents LREM
    "UDI-I": "UDI",
    "RN": "RN",
    "RE": "LREM",
    "LFI": "LFI",
    "LIOT": "LIOT",
    "SOC-A": "SOC",
    "ECOLO": "ECOLO",
    "HOR": "HOR",  # ??? dissidents AGIR-E et LREM
    "GDR-NUPES": "GDR",
    "EPR": "LREM",
    "ECOS": "ECOLO",
    "DR": "LR",
    "UDDPLR": "UDR",
    "UDR": "UDR",
    "LFI-NUPES": "LFI",
    "LFI-NFP": "LFI",
}


def err(*args):
    print(*args, file=sys.stderr)


def _is_gouvernement(function):
    for fn in [
        "ministre",
        "secrétaire d'état",
        "secrétaire d’état",
        "garde des sceaux",
    ]:
        if function and fn in function.lower():
            return True
    return False


# Heuristique de normalisation des abbréviations de groupe
# différentes entre les didascalies et les interruptions
def _normalize_grp(string):
    return string.upper()


def _harmonized_grp(string):
    return HARMONIZED_GROUPS.get(string, None)


def fuzz_search(string, refs):
    if not refs:
        return None
    scores = Counter({occ: ratio(string, occ) for occ in refs})
    mc, sc = scores.most_common(1)[0]
    return mc if sc > 80 else None


def prepare_binding(data):
    # Path : date > dossier > amendment number
    items = defaultdict(
        lambda: {
            "dossiers": set(),
            "amendments": defaultdict(lambda: defaultdict(lambda: None)),
            "rejets": defaultdict(lambda: defaultdict(lambda: None)),
            "articles": defaultdict(lambda: defaultdict(lambda: None)),
            "ensemble": defaultdict(lambda: defaultdict(lambda: None)),
        }
    )

    for item in data.values():
        date = item["date"]
        id = item["id"]
        dossier = ebbe.getpath(item, ["dossier", "title"])
        title = item["title"]

        if not dossier:
            continue

        items[date]["dossiers"].add(dossier)

        # Si on rencontre une motion de rejet préalable
        # on la traite puis on continue
        if MOTION_REJET.search(title):

            group = {
                "found": True,
                "id": id,
                "dossier": dossier,
                "title": title,
            }

            items[date]["rejets"][dossier] = group.copy()
            
            continue

        # Si on rencontre un vote sur article
        # on le traite puis on continue
        if ARTICLE_VOTE.search(title):
            art = ARTICLE_VOTE.search(title)
            
            num = ''.join(art.captures("num"))
            sup = ''.join(art.captures("sup"))
            let = ''.join(art.captures("letter"))

            num = "1" if num == "unique" or num == "premier" or num == "1er" else num
            
            label = ' '.join(a for a in [num, sup, let] if a)

            group = {
                "found": True,
                "id": id,
                "dossier": dossier,
                "title": title,
                "article": art,
                "label": label
            }

            items[date]["articles"][dossier][label] = group.copy()
            items[date]["articles"][dossier][art] = group.copy()

            continue

        if ENSEMBLE.search(title):
            group = {
                "found": True,
                "id": id,
                "dossier": dossier,
                "title": title,
            }

            items[date]["ensemble"][dossier] = group.copy()

            continue

        amdts = set()
        suivant = False
        for m in AMENDEMENT.finditer(title):
            with_target = bool(m.captures("parent"))

            if with_target:
                continue

            for a in m.captures("number"):
                amdts.add(a)

            if m.captures("suivant"):
                suivant = True

        group = {
            "found": True,
            "id": id,
            "dossier": dossier,
            "title": title,
            "amendments": amdts,
            "suivant": suivant,
        }

        for amd in amdts:
            instance = items[date]["amendments"][dossier][amd]

            if isinstance(instance, dict):
                continue
            else:
                items[date]["amendments"][dossier][amd] = group.copy()

    return items


def open_amendments(filepath):

    # Path : date > dossier > amendment number
    items = defaultdict(
        lambda: {
            "dossiers": set(),
            "amendments": defaultdict(lambda: defaultdict(lambda: None)),
        }
    )

    with casanova.reader(filepath) as reader:
        for row in reader:
            date, _, dossier, number, author_name, author_group, content, summary = row

            items[date]["dossiers"].add(dossier)

            # TODO vérifier que ça fonctione....
            # Permet d'enlever les "rectifié"
            number = number.split(' ')[0]

            group = {
                "found": True,
                "number": number,
                "dossier": dossier,
                "author_name": author_name,
                "author_group": author_group,
                "content": content,
                "summary": summary,
            }

            instance = items[date]["amendments"][dossier][number]
            if isinstance(instance, dict):
                raise Exception(f"shouldn't exist : ({date}) {instance} {group}")
            else:
                items[date]["amendments"][dossier][number] = group.copy()

    return items


"""
Method to get the ID of the vote issue 
related to the speech (if a vote occured !)
"""
def get_vote_ids(row, binding):
    date, subject, s_subject, ss_subject, amdts, amdts_p = row[3], row[5], row[6], row[14], row[15], row[16]

    if RAPPEL_REGLEMENT.search(ss_subject):
        return [], False, None

    amdts = amdts.split("|") if amdts else []
    amdts_p = amdts_p.split("|") if amdts_p else []

    votes_dossiers = binding[date]["dossiers"]
    vote_dossier = fuzz_search(subject, votes_dossiers)

    if not vote_dossier:
        return [], False, None
    
    # Checking MOTION REJET

    if MOTION_REJET.search(s_subject):
        if v := binding[date]["rejets"][vote_dossier]:
            return [v["id"]], True, "motion_rejet"


    # Checking EXPLICATION VOTES

    if EXPLICATION_VOTE.search(s_subject):
        if v := binding[date]["ensemble"][vote_dossier]:
            return [v["id"]], True, "explication_vote"
    
    # Checking AMENDMENTS

    if amdts:
        found = []
        amd_p_bool = False
        if amdts_p:
            for amd in amdts_p:
                v = binding[date]["amendments"][vote_dossier][amd]
                if v:
                    found.append(v)
                    amd_p_bool = True
        else:
            for amd in amdts:
                v = binding[date]["amendments"][vote_dossier][amd]
                if v:
                    found.append(v)

        # Rule : if an amendment_precise isn't found
        # BUT vote label indicates "suivant", we consider
        # this amendment_precise as a part of "suivant"
        # we don't know for sure if it's true BUT we assume it
        # => to investigate further in the data, to check
        #    if it's true
        # However, because it's unsure, we don't mark amd_p_bool
        # as True
        if amdts_p and not found:
            for amd in amdts:
                v = binding[date]["amendments"][vote_dossier][amd]
                if v and v["suivant"]:
                    found.append(v)

        if found:
            return [v["id"] for v in found], amd_p_bool, "amendment"
    
   
    else:
        if art := ARTICLE.search(ss_subject):
            pos = ''.join(art.captures("pos"))
            num = ''.join(art.captures("num"))
            sup = ''.join(art.captures("sup"))
            let = ''.join(art.captures("letter"))

            if pos: 
                return [], False, None

            label = ' '.join([a for a in [pos, num, sup, let] if a])

            if v := binding[date]["articles"][vote_dossier][label]:
                return [v["id"]], True, "article"

    
    return [], False, None
    

"""
Method to get the vote of a speaker
when provided a vote id
"""
def get_amendments(row, binding):
    date, subject, amdts, amdts_p = row[3], row[5], row[15], row[16]

    amdts = amdts.split("|") if amdts else []
    amdts_p = amdts_p.split("|") if amdts_p else []

    votes_dossiers = binding[date]["dossiers"]
    vote_dossier = fuzz_search(subject, votes_dossiers)
    if not vote_dossier:
        return []

    found = []
    if amdts_p:
        for amd in amdts_p:
            v = binding[date]["amendments"][vote_dossier][amd]
            if v:
                found.append(v)
    else:
        for amd in amdts:
            v = binding[date]["amendments"][vote_dossier][amd]
            if v:
                found.append(v)

    return [
        (v["number"], v["author_name"], v["author_group"], v["content"], v["summary"])
        for v in found
    ]


def get_vote_issue(votes, row, vote_id):
    name, group, last_group, function = row[7], row[9], row[10], row[11]

    if function == "président":
        return ""
    
    vote = votes.from_name(vote_id, name)

    if not vote and not group:
        vote = votes.from_group(vote_id, last_group)
        if not vote:
            for g in GROUPS_GOVERNMENT.get(name, MAJORITY_GROUPS):
                vote = votes.from_group(vote_id, g)
                if vote:
                    break
        if not vote:
            for g in MAJORITY_GROUPS:
                vote = votes.from_group(vote_id, g)
                if vote:
                    break

    return vote



with open(VOTES_JSON) as file:
    data = json.load(file)
    votes = Votes(VOTES_JSON)

    binding_votes = prepare_binding(data)

#binding_amendments = open_amendments(AMENDMENTS_CSV)

with open(INPUT_CSV) as source:
    enricher = casanova.enricher(
        source,
        sys.stdout,
        add=[
            "is_government",
            "vote_id",
            "vote_issue",
            "vote_type"
        ],
    )
    for row in enricher:
        date, function, subject, amendments, amendments_precise = (
            row[3],
            row[11],
            row[5],
            row[15],
            row[16],
        )

        vote_ids, precised_vote, vote_type = get_vote_ids(row, binding_votes)
        votes_issues = list(map(lambda a: str(get_vote_issue(votes, row, a)), vote_ids))

        #amendments = get_amendments(row, binding_amendments)

        row[9] = _harmonized_grp(_normalize_grp(row[9]))

        gov = _is_gouvernement(function)

        str_vote_ids = "|".join(vote_ids)
        str_vote_issues = "|".join(votes_issues)

        enricher.writerow(
            row,
            [
                gov,
                str_vote_ids,
                str_vote_issues,
                vote_type          
            ],
        )