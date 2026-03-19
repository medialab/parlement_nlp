import glob
import re
import bs4
import tqdm
import stanza
import casanova

from lib.actors import Actors
from datetime import datetime as dt
from pathlib import Path
from os.path import join

ID = re.compile(r'\/([A-Za-z0-9]+?)\.json')
ITALIC = re.compile(r'<italique>.*?\((.*?)\).*?</italique>')
TAG = re.compile(r'<.+?>')
DIDASCALIES_SPLITTER_RE = re.compile(r'[.\s]–')
TODAY = dt.today().strftime('%Y-%m-%d')

MEMES_MOUVEMENTS_RE = re.compile('mêmes?\s+mouvements?')
CIVIL_RE = re.compile(r'((MM?\.)|(Mmes?))')

QUICK = False

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

HEADERS = [
    "intervention_id",
    "debate_number",
    "debate_ref",
    "date",
    "moment",
    "subject",
    "sub_subject",
    "name",
    "sexe",
    "group",
    "last_group",
    "function",
    "code",
    "intervention"
]

LEGIS_NUM = re.compile(r'([0-9]{2})th')

class DebateBuilder:
    def __init__(self, dir, output_path, quick=False, format="csv"):
        self.quick = quick or format == "csv"

        print("Loading actors...")
        self.acteurs = Actors()
        self.legis_num = LEGIS_NUM.search(dir).group(1)

        if not self.quick:
            self.nlp = stanza.Pipeline('fr')

        writer = casanova.writer(output_path, HEADERS)

        print("Treating debates...")
        paths = glob.glob(join(dir, "*.xml"))
        
        i = 0
        for path in tqdm.tqdm(paths, total=len(paths)):
            content = Path(path).read_text()
            soup = bs4.BeautifulSoup(content, "xml")
            _, intervs_list = self._get_intervs(soup)

            for row in intervs_list:
                writer.writerow([str(f"{self.legis_num}-{i}")] + list(row))
                i += 1

        writer.close()

    def _extract_didascalie(self, texte: str):
        # Récupération des didascalies en italique
        results = []
        offset = 0
        for it in ITALIC.finditer(texte):
            before = TAG.sub(' ', texte[offset:it.start()]).strip()
            if before:
                results.append((before, False))

            # Splitting des didascalies multiples
            # Cf. https://github.com/medialab/didascalies/blob/master/scripts/clean_didascalies.py
            didascalies = it.group(1)
            didascalies = TAG.sub('', didascalies)
            didascalies = [d.strip('. ') for d in DIDASCALIES_SPLITTER_RE.split(didascalies.strip('. '))]

            for d in didascalies:
                results.append((d, True))
            offset = it.end()

        last = TAG.sub(' ', texte[offset:len(texte)]).strip()
        if last: results.append((last, False))

        return results
    
    def _extract_speaker(self, element, date):
        parent = element.parent
        speaker = parent.select_one("orateurs orateur")

        if not speaker:
            return None, None, None, None, None

        name = speaker.select_one("nom")
        try:
            id = "PA" + speaker.select_one("id").text
        except:
            return None, None, None, None, None

        qualite = speaker.select_one("qualite")

        actor = self.acteurs.get(id, date)
        if not actor:
            return name.text, None, None, None, None

        function = "président" if "président" in name.text else qualite.text
        name = actor["name"]
        sexe = actor["sexe"]
        group = actor["group"]
        last_group = actor["last_group"]

        return name, sexe, group, last_group, function
    
    def _get_code(self, texte):
        code = texte.parent.attrs["code_grammaire"]
        if "PAROLE_" in code:
            return "PAROLE_GENERIQUE"
        elif "INTERRUPTION" in code:
            return "INTERRUPTION"
        else:
            return None
        
    # Pas très propre mais la date time ISO
    # du XML semble régulière...
    def _format_date(self, string):
        year = string[0:4]
        month = string[4:6]
        day = string[6:8]
        return f"{year}-{month}-{day}"
    
    def _is_government(self, function):
        for fn in ["ministre", "secrétaire d'état", "secrétaire d’état", "garde des sceaux"]:
            if function and fn in function.lower():
                return True
        return False
    
    def _parse_didascalie(self, string, date):
        if self.quick:
            return [], [], [], string

        groups = []
        for grp in GROUPS:
            if grp in string:
                groups.append(self._normalize_grp(grp))

        # Identification des personnes
        # en limitant l'application du moteur NER
        # sur les chaînes contenant M. Mme Mmes et MM.

        persons_parsed = []
        if CIVIL_RE.search(string):
            doc = self.nlp(string)
            for en in doc.ents:
                if en.type != "PER": continue
                persons_parsed.append(CIVIL_RE.sub('', en.text).strip())

        persons = []
        if persons_parsed:
            for pers in persons_parsed:
                obj = self.acteurs.search(pers, date)
                if obj:
                    persons.append(obj["name"])
                    groups.append(self._normalize_grp(obj["group"]))
        
        actions = []
        sentiment = 0
        for pos_action in POSITIVE_KEY_WORDS.keys():
            if pos_action in string.lower():
                actions.append(POSITIVE_KEY_WORDS[pos_action])
                sentiment += 1
                break
        for neg_action in NEGATIVE_KEY_WORDS.keys():
            if neg_action in string.lower():
                actions.append(NEGATIVE_KEY_WORDS[neg_action])
                sentiment -=1
                break

        if sentiment == 0:
            sentiment = "NEUTRE"
        elif sentiment > 0:
            sentiment = "POSITIF"
        else:
            sentiment = "NEGATIF"

        return groups, persons_parsed, actions, sentiment
    
    def _normalize_grp(self, string):
        return string.upper()
    
    def _get_intervs(self, soup):
        interventions = []
        date = soup.select_one("compteRendu metadonnees dateSeance").text
        date = self._format_date(date)

        debate_id = soup.select_one("compteRendu metadonnees numSeance").text
        debate_ref = soup.select_one("compteRendu seanceRef")
        if debate_ref:
            debate_ref = debate_ref.text
        else:
            debate_ref = None

        
        subject = ""
        sub_subject = ""
        for text in soup.select("compteRendu texte"):
            # Verification parent == point
            # auquel cas on garde le sujet
            if text.parent.name == "point":
                if text.parent.attrs["nivpoint"] == "1":
                    subject = text.get_text()
                if text.parent.attrs["nivpoint"] == "2":
                    sub_subject = text.get_text()

            moment = text.attrs.get("stime")
            code = self._get_code(text)

            name, sexe, group, last_group, function = self._extract_speaker(text, date)
            
            # Règle unique
            if code != "INTERRUPTION" and (name and sexe):
                code = "PAROLE_GENERIQUE"

            interventions_and_didascalies = self._extract_didascalie(str(text))

            for (interv, didascalie) in interventions_and_didascalies:
                if didascalie:
                    interventions.append((
                        debate_id,
                        debate_ref,
                        date,
                        moment,
                        subject,
                        sub_subject,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "DIDASCALIE",
                        interv
                    ))
                else:
                    interventions.append((
                        debate_id,
                        debate_ref,
                        date,
                        moment,
                        subject,
                        sub_subject,
                        name,
                        sexe,
                        group,
                        last_group,
                        function,
                        code,
                        interv
                    ))
        
        return debate_ref, interventions
