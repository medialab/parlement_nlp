import casanova
import re
import json

from datetime import datetime as dt

import argparse

# === ARGUMENTS ===

parser = argparse.ArgumentParser(description="Regroupement of debates")
parser.add_argument("--segments-csv", type=str, help="CSV path of relevant segments")
parser.add_argument("--output-json", type=str, help="Path of the JSON output")
args = parser.parse_args()

SEGMENTS_CSV = args.segments_csv
OUTPUT_JSON = args.output_json

# === CONSTANTS ===

ITALIC = re.compile(r'<italique>.*?\((.*?)\).*?</italique>')
TAG = re.compile(r'<.+?>')
DIDASCALIES_SPLITTER_RE = re.compile(r'[.\s]–')
TODAY = dt.today().strftime('%Y-%m-%d')

MEMES_MOUVEMENTS_RE = re.compile('mêmes?\s+mouvements?')
CIVIL_RE = re.compile(r'((MM?\.)|(Mmes?))')

GROUPS = [
    "LR",
    "RN",
    "LFI-NUPES",
    "Écolo-NUPES",
    "Dem",
    "SOC",
    "GDR-NUPES",
    "LIOT",
    "RE",
    "HOR",
    "EcoS",
    "DR",
    "LFI-NFP",
    "SOC",
    "EPR",
    "GDR",
    "UDR"
]

BLACK_LIST = [
    'article',
    'motion',
    'séance',
    'seance',
    'amendement',
    'projet',
    'scrutin',
    'proposition',
    'votants',
    'remplace',
    'adopté',
    'tirage au sort'
]

# Adding tonality marker
POSITIVE_KEY_WORDS = {
  'applaud': 'applaudissement',
  'sourir': 'sourir',
  'oui': 'oui',
  'approb': 'approbation',
  'approuv': 'approuvement',
  'rir': 'rire',
  'rit': 'rire',
  'assentiment': 'assentiment'
}

NEGATIVE_KEY_WORDS = {
  'protest': 'protestation',
  'non': 'non',
  'exclam': 'exclamation',
  'rappel au règlement': 'rappel au règlement',
  'rappels au règlement': 'rappel au règlement',
  'interrupt': 'interruption',
  'scandal': 'scandale',
  'oh': 'oh',
  'dénégat': 'dénégation',
  'faux': 'faux',
  'stop': 'stop',
  'hué': 'hué',
  'brouhaha': 'brouhaha',
  'sifflement': 'sifflement',
  'tumult': 'tumult'
}

HARMONIZED_GROUPS = {
    "FI":       "LFI",
    "LAREM":    "LREM",
    "SOC":      "SOC",
    "UDI-AGIR": "UDI",
    "GDR":      "GDR",
    "MODEM":    "MODEM",
    "NG":       "SOC",
    "LR":       "LR",
    "LT":       "LIOT",
    "AGIR-E":   "AGIR-E", # ??? dissidents LREM et UDI
    "UDI_I":    "UDI",
    "NI":       "NI",
    "UDI-A-I":  "UDI",
    "LC":       "UDI",
    "DEM":      "MODEM",
    "EDS":      "EDS", # ??? dissidents LREM
    "UDI-I":    "UDI",
    "RN":       "RN",
    "RE":       "LREM",
    "LFI":      "LFI",
    "LIOT":     "LIOT",
    "SOC-A":    "SOC",
    "ECOLO":    "ECOLO",
    "HOR":      "HOR", # ??? dissidents AGIR-E et LREM
    "GDR-NUPES": "GDR",
    "EPR":      "LREM",
    "ECOS":     "ECOLO",
    "DR":       "LR",
    "UDDPLR":   "UDR",
    "UDR":      "UDR",
    "LFI-NUPES":"LFI",
    "LFI-NFP":  "LFI"
}

def group(rows):
    groups = []

    current_id = ""
    current_grp = []
    for row in rows:
        scrutin = row[0]

        if current_id != scrutin:
            if current_grp: groups.append(current_grp)
            
            current_id = scrutin
            current_grp = []

        current_grp.append(row)
    
    groups.append(current_grp)

    return groups

def _is_gouvernement(function):
    for fn in ["ministre", "secrétaire d'état", "secrétaire d’état", "garde des sceaux"]:
        if function and fn in function.lower():
            return True
    return False

# Heuristique de normalisation des abbréviations de groupe
# différentes entre les didascalies et les interruptions
def _normalize_grp(string):
    return string.upper()

def _harmonized_grp(string):
    return HARMONIZED_GROUPS.get(string, None)

def to_json(rows):    
    root = {
        "vote_id": None,
        "amendment": None,
        "subject": None,
        "debate_ref": None,
        "date": None,
        "turns": [],
    }

    parole = None
    temp = None
    for i, row in enumerate(rows):
        vote_instance, amendment, author_name, author_group, amendment_content, amendment_summary, _, _, _, _, vote_issue, intervention_id, _, debate_ref, date, moment, subject, sub_subject, name, sexe, group, _, function, code, intervention = row

        root["vote_id"] = vote_instance
        root["amendment"] = amendment
        root["author_name"] = author_name
        root["author_group"] = author_group
        root["content"] = amendment_content
        root["summary"] = amendment_summary
        root["subject"] = subject
        root["debate_ref"] = debate_ref
        root["date"] = date

        if code == "PAROLE_GENERIQUE":
            # Règle
            # Si la parole générique est celle d'un parlementaire
            # différent de celui d'avant et si ce n'est pas la/le
            # président-e de séance, alors on crée un nouvel objet
            # intervention (temp)
            if name != parole:                
                temp = {
                    "type": "speech",
                    "moment": moment,
                    "sub_subject": sub_subject,
                    "speaker": {
                        "name": name,
                        "sexe": sexe,
                        "group": _normalize_grp(group),
                        "group_harmonized": _harmonized_grp(group),
                        "government": _is_gouvernement(function),
                        "function": function,
                    },
                    "vote": vote_issue,
                    "speech": []
                }

                root["turns"].append(temp.copy())
                
                parole = name

            
            temp["speech"].append({
                "speech_id": intervention_id,
                "speech": intervention,
                "reactions": []
            })

        if code == "DIDASCALIE":
            # TODO rajouter les didascalies
            continue

            # On évacue les "fausses" didascalies
            # avec une petite heuristique
            # pour enlever les intervention
            # vraiment trop courte
            stop = False
            for keyword in BLACK_LIST:
                if keyword in intervention:
                    stop = True
            if stop: continue

            if len(intervention.split(' ')) < 3: continue

            groupes, persons, actions, = _parse_didascalie(intervention, date)

            try:
                temp["interventions"][-1]["reactions"].append({
                    "type": "didascalie",
                    "moment": moment,
                    "groupes": groupes,
                    "personnes": persons,
                    "actions": actions,
                    "intervention": intervention
                })
            except:
                #print(nom, code, intervention)
                pass


        if code == "INTERRUPTION":
            if not temp: continue

            temp["speech"][-1]["reactions"].append({
                "type": "interruption",
                "moment": moment,
                "speaker": {
                    "name": name,
                    "sexe": sexe,
                    "group": _normalize_grp(group),
                    "group_harmonized": _harmonized_grp(group),
                    "government": _is_gouvernement(function),
                    "function": function,
                },
                "vote": vote_issue,
                "speech_id": intervention_id,
                "speech": intervention
            })
        
        if code != "PAROLE_GENERIQUE" and code != "DIDASCALIE" and function == "président":
            #On vérifie que le dernier caractère est une fin de phrase
            if intervention.strip()[-1] not in ('.', '!', '?', '…'):
                continue

            if name != parole:                
                temp = {
                    "type": "speech",
                    "moment": moment,
                    "sub_subject": sub_subject,
                    "speaker": {
                        "name": name,
                        "sexe": sexe,
                        "group": _normalize_grp(group),
                        "group_harmonized": _harmonized_grp(group),
                        "government": _is_gouvernement(function),
                        "function": function,
                    },
                    "vote": vote_issue,
                    "speech": []
                }

                root["turns"].append(temp.copy())
                
                parole = name
            
            temp["speech"].append({
                "speech_id": intervention_id,
                "speech": intervention,
                "reactions": []
            })

    return root

rows = list(casanova.reader(SEGMENTS_CSV))
groups = group(rows)
groups = [to_json(rows) for rows in groups]

with open(OUTPUT_JSON, "w") as export:
    json.dump(groups, export, ensure_ascii=False)