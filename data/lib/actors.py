import glob
import re
import json

from pathlib import Path
from thefuzz.fuzz import ratio
from collections import Counter

from datetime import datetime as dt

PATH_ORGANE = "./open_data/actors/organe/*.json"
PATH_ACTEUR = "./open_data/actors/acteur/*.json"

ID = re.compile(r'\/([A-Za-z0-9]+?)\.json')

TODAY = dt.today().strftime('%Y-%m-%d')

class Actors:
    def __init__(self):

        docs_organe = glob.glob(PATH_ORGANE)
        docs_acteur = glob.glob(PATH_ACTEUR)

        self.docs_organe = { p: json.loads(Path(p).read_text()) for p in docs_organe }
        self.docs_acteur = { p: json.loads(Path(p).read_text()) for p in docs_acteur }

        self.date = None

    def _reload_data(self, date):
        if self.date == date:
            return
        
        self.date = date
        
        organes = {}
        for p in self.docs_organe.keys():
            id = ID.search(p).group(1)
            data = self.docs_organe[p]
            organes[id] = data["organe"]["libelleAbrev"]

        self.actors = {}
        self.id_name = {}
        for p in self.docs_acteur:
            id = ID.search(p).group(1)
            data = self.docs_acteur[p]

            first_name = data["acteur"]["etatCivil"]["ident"]["prenom"]
            last_name = data["acteur"]["etatCivil"]["ident"]["nom"]
            sexe = "F" if data["acteur"]["etatCivil"]["ident"]["civ"] == "Mme" else "H"
            mandats = data["acteur"]["mandats"]["mandat"]

            current_group = None
            last_group, last_date = None, "1997-01-01"

            mandats = mandats if type(mandats) == list else [mandats]
            for m in mandats:
                try:
                    if m["typeOrgane"] == "GP":
                        org = m["organes"]["organeRef"]

                        start = m["dateDebut"]
                        end = m["dateFin"]

                        if not end:
                            end = TODAY


                        if start <= self.date <= end:
                            current_group = org

                        if end > last_date and start <= self.date:
                            last_group = org
                            last_date = end


                except Exception:
                    pass


            self.actors[f"{first_name} {last_name}"] = {
                "id": id,
                "group": organes[current_group] if current_group else None,
                "last_group": organes[last_group] if last_group else None,
                "name": f"{first_name} {last_name}",
                "sexe": sexe
            }

            
            self.id_name[id] = f"{first_name} {last_name}"

    def search(self, string, date, min_ratio=80):
        self._reload_data(date)

        candidates = Counter()

        for name in self.actors.keys():
            r = ratio(string, name)
            if r >= min_ratio:
                candidates[name] = r

        if len(candidates) > 0:
            key = candidates.most_common(1)[0][0]
            return self.actors[key] # id, groupe, nom, sexe

        return None
    
    def get(self, id, date):
        self._reload_data(date)

        pn = self.id_name.get(id)
        if not pn:
            return None
        
        return self.actors.get(pn)
