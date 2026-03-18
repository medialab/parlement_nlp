import glob
import json
import ebbe
import tqdm
import sys

from os.path import join
from pathlib import Path
from collections import Counter
from thefuzz.fuzz import ratio

from lib.dossiers import Dossiers
from lib.actors import Actors
import argparse

parser = argparse.ArgumentParser(description="Parse scrutins and output as JSON.")
parser.add_argument("--votes-dir", help="Directory containing scrutins JSON files")
parser.add_argument("--dossiers-dir", help="Directory containing dossiers JSON files")
args = parser.parse_args()

SCRUTINS_DIR = args.votes_dir
DOSSIERS_DIR = args.dossiers_dir

dossiers = Dossiers(DOSSIERS_DIR)
actors = Actors()

def parse_vote(path):
    data = json.loads(Path(path).read_text())
    date = ebbe.getpath(data, ["scrutin", "dateScrutin"])
    vote_title = ebbe.getpath(data, ["scrutin", "titre"])
    debate_ref = ebbe.getpath(data, ["scrutin", "seanceRef"])
    vote_id = ebbe.getpath(data, ["scrutin", "uid"])
    
    # parsing seance
    dossier = dossiers.from_seance(debate_ref, vote_title)

    all_votes = []
    summary = {
        "POUR": 0,
        "CONTRE": 0,
        "ABSTENTION": 0
    }
    groups = ebbe.getpath(data, ["scrutin", "ventilationVotes", "organe", "groupes", "groupe"])
    for group in groups:
        votes = ebbe.getpath(group, ["vote", "decompteNominatif"])
        
        pours = ebbe.getpath(votes, ["pours", "votant"], [])
        contres = ebbe.getpath(votes, ["contres", "votant"], [])
        abstentions = ebbe.getpath(votes, ["abstentions", "votant"], [])

        count = [
            ("POUR", pours),
            ("CONTRE", contres),
            ("ABSTENTION", abstentions)
        ]
        for label, speakers in count:
            if type(speakers) == dict:
                speakers = [speakers]

            for speaker in speakers:
                actor_id = speaker["acteurRef"]
                actor = actors.get(actor_id, date)

                nom = actor["name"]
                grp = actor["group"]
                sexe = actor["sexe"]
               
                
                all_votes.append({
                    "name": nom,
                    "group": grp,
                    "sexe": sexe,
                    "vote": label
                })

                summary[label] += 1

    return {
        "date": date,
        "id": vote_id,
        "debate": debate_ref,
        "title": vote_title,
        "dossier": {
            "id": dossier["id"],
            "title": dossier["title"],
        } if dossier else None,
        "summary": summary,
        "votes": all_votes
    }
    

def filtre(scrutin):
    titre = scrutin["title"]

    if "ensemble" not in titre.lower():
        return False
    
    if "proposition" not in titre.lower():
        return False
        
    #print(scrutin["synthese"])
     # Changer pour prendre en compte ABSTENTION
    part_pour = scrutin["summary"]["POUR"] / (scrutin["summary"]["POUR"] + scrutin["summary"]["CONTRE"])
    part_contre = scrutin["summary"]["CONTRE"] / (scrutin["summary"]["POUR"] + scrutin["summary"]["CONTRE"])

    if part_pour < 0.25 or part_contre < 0.25:
        return False
    
    return True

def argmax(items, good):
    count = Counter({ i: ratio(i, good) for i in items })
    return count.most_common(1)[0][0]

def get_debate_part(vote, debates):
    dossier_title = vote["dossier"]["title"]
    compte_rendu = debates[vote["debate"]]
    subjects = [item["subject"] for item in compte_rendu]

    subject_target = argmax(subjects, dossier_title)

    return [item for item in compte_rendu if item["subject"] == subject_target]

    
def list_votes(dir, filter=None):
    items = {}
    paths = glob.glob(join(dir, "*.json"))
    for path in tqdm.tqdm(paths, total=len(paths)):
        vote = parse_vote(path)
        id = vote["id"]
        
        if filter and not filter(vote):
            continue

        items[id] = vote

    return items


votes = list_votes(SCRUTINS_DIR)
json.dump(votes, sys.stdout, ensure_ascii=False)