import casanova
import re
import tqdm

from lib.votes import Votes
from lib.amendments import Amendments
from lib.dossiers import Dossiers
import argparse

parser = argparse.ArgumentParser(description="Parse votes and amendements")
parser.add_argument("--output", help="Output CSV file path")
parser.add_argument("--votes", help="Votes JSON file path")
parser.add_argument("--amendments", help="Amendements CSV file path")
parser.add_argument("--dossiers", help="Dossiers directory path")

args = parser.parse_args()

OUTPUT = args.output
VOTES_JSON = args.votes
AMENDMENTS_CSV = args.amendments
DOSSIERS_DIR = args.dossiers

votes = Votes(VOTES_JSON)
amendments = Amendments(AMENDMENTS_CSV)
dossiers = Dossiers(DOSSIERS_DIR)

AMENDEMENT = re.compile(r'amendement.+?n°\s*?(\w+)')

def extract_amendment(vote_title):
    return AMENDEMENT.search(vote_title).group(1)

def filtre(vote):
    date = vote["date"]
    vote_title = vote["title"]
    if not vote["dossier"]: return False
    
    dossier = vote["dossier"]["title"].lower()

    # On enlève les lois de finance
    if "loi de finance" in dossier:
        return False
    
    # On enlève les scrutins sur des amendements multiples
    if "amendements identiques" in vote_title:
        return False
    
    if "amendement identique" in vote_title:
        return False
    
    nb_amendments = len(AMENDEMENT.findall(vote_title))

    if nb_amendments != 1:
        return False
    
    return date, vote_title, dossier

with casanova.writer(OUTPUT, ["date", "dossier", "vote_title", "vote_id", "amendment", "author_name", "author_group", "amendment_content", "amendment_summary"]) as writer:
    for scrutin in tqdm.tqdm(votes.filter(filtre)):
        date = scrutin["date"]
        id = scrutin["id"]
        debate = scrutin["debate"]
        vote_title = scrutin["title"]
        dossier = scrutin["dossier"]["title"]
        bills = dossiers.get(scrutin["dossier"]["id"])["bills"]

        amendment = extract_amendment(vote_title)
        
        author_name = None
        author_group = None
        amendment_content = None
        amendment_summary = None
        
        objs = amendments.get(date, amendment)
        
        if objs:
            stop = False
            for obj in objs:
                for pd in dossiers.from_seance(obj["debate"]):
                    pd_title = pd["title"]
                    if pd_title == dossier:
                        author_name = obj["author_name"]
                        author_group = obj["author_group"]
                        amendment_content = obj["amendment_content"]
                        amendment_summary = obj["amendment_summary"]
                        stop = True
                    if stop: break
                if stop: break

        if not amendment_content and not amendment_summary:
            continue

        writer.writerow([
            date, dossier, vote_title, id, amendment, author_name, author_group, amendment_content, amendment_summary
        ])