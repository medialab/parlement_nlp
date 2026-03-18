import glob
import json
import ebbe

from thefuzz.fuzz import ratio
from collections import defaultdict, Counter
from pathlib import Path
from os.path import join, split

class Dossiers:

    def _actes_legis_rec(self, obj, attribute=[]):
        if not obj: return []
        children = ebbe.getpath(obj, ["actesLegislatifs", "acteLegislatif"])
        if not children: return []
        if type(children) == dict: children = [children]

        results = []
        for child in children:
            for attr in attribute:
                if attr in child and type(child) == dict:
                    #results.append(child[attr])
                    if "dateActe" in child:
                        results.append({"id": child[attr], "date": child["dateActe"][:10]})
            if "actesLegislatifs" in child:
                try:
                    results += self._actes_legis_rec(child, attribute)
                except Exception as e:
                    #print(obj)
                    raise e

        return results

    def __init__(self, dossiers_dir):
        self.dossiers = {}
        self.debates = defaultdict(list)
        self.dates = defaultdict(list)
        for path in glob.glob(join(dossiers_dir, "*.json")):
            data = json.loads(Path(path).read_text())

            _, file = split(path)

            id = file.replace('.json', '')

            title = ebbe.getpath(data, ["dossierParlementaire", "titreDossier", "titre"])
            label = ebbe.getpath(data, ["dossierParlementaire", "procedureParlementaire", "libelle"])
            debates_id = self._actes_legis_rec(ebbe.get(data, "dossierParlementaire"), ["reunionRef"])
            bills = self._actes_legis_rec(ebbe.get(data, "dossierParlementaire"), ["texteAssocie"])

            dossier = {
                "id": id,
                "title": title,
                "label": label,
                "debates": debates_id,
                "bills": bills
            }

            self.dossiers[id] = dossier
            for sdict in debates_id:
                self.debates[sdict["id"]].append(dossier)
                self.dates[sdict["date"]].append(dossier)


    def get(self, id):
        return self.dossiers[id]
    
    def from_seance(self, id, clue = None):
        debates = self.debates.get(id, [])

        if not clue:
            return debates
        
        titles = { t["title"]: i for i, t in enumerate(debates) }

        prox = Counter({ t: ratio(clue, t) for t in titles })
        try:
            mt = prox.most_common(1)[0][0]
            return debates[titles[mt]]
        except Exception as e:
            return None
        
    def from_date(self, date, clue):
        debates = self.dates.get(date, [])

        if not clue:
            return debates
        
        titles = { t["title"]: i for i, t in enumerate(debates) }

        prox = Counter({ t: ratio(clue, t) for t in titles })
        try:
            mt = prox.most_common(1)[0][0]
            return debates[titles[mt]]
        except Exception as e:
            return None