import glob
import json
import ebbe
import bs4
import casanova
import tqdm
import os

from os.path import join

from lib.actors import Actors
from lib.dossiers import Dossiers
import argparse

parser = argparse.ArgumentParser(description="Build amendments CSV from JSON files.")
parser.add_argument("--dossiers-dir", help="Directory containing dossiers JSON files")
parser.add_argument(
    "--amendments-dir", help="Directory containing amendments JSON files"
)
parser.add_argument("--output-csv", help="Output CSV file path")
args = parser.parse_args()

DOSSIERS_DIR = args.dossiers_dir
AMENDEMENTS_DIR = args.amendments_dir
OUTPUT_CSV = args.output_csv

actors = Actors()
dossiers = Dossiers(DOSSIERS_DIR)

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

if __name__ == "__main__":
    paths = list(glob.glob(join(AMENDEMENTS_DIR, "*/*/*.json")))

    with casanova.writer(
        OUTPUT_CSV,
        [
            "date",
            "debate",
            "dossier",
            "number",
            "author_name",
            "author_group",
            "amendment_content",
            "amendment_summary",
        ],
    ) as writer:
        for path in tqdm.tqdm(paths):
            with open(path) as source:
                data = json.load(source)

            # dossier
            dossier_id = path.split("/")[-3]
            dossier = dossiers.get(dossier_id)
            if not dossier:
                continue
            dossier = dossier["title"]

            # text
            bill = ebbe.getpath(data, ["amendement", "texteLegislatifRef"])
            if not bill or not isinstance(bill, str):
                continue

            # debate
            debate = ebbe.getpath(data, ["amendement", "seanceDiscussionRef"])
            if not debate or not isinstance(debate, str):
                continue

            # number
            number = ebbe.getpath(data, ["amendement", "identification", "numeroLong"])
            if not number or not isinstance(number, str):
                continue

            # date sort
            date = ebbe.getpath(data, ["amendement", "cycleDeVie", "dateSort"])
            if not date or not isinstance(date, str):
                continue
            date = date[:10]

            # dispositif
            content = ebbe.getpath(
                data, ["amendement", "corps", "contenuAuteur", "dispositif"]
            )
            summary = ebbe.getpath(
                data, ["amendement", "corps", "contenuAuteur", "exposeSommaire"]
            )

            # depute
            author = ebbe.getpath(
                data, ["amendement", "signataires", "auteur", "acteurRef"]
            )
            if not author or not isinstance(author, str):
                continue
            author = actors.get(author, date)

            if author:
                author_name = author["name"]
                author_group = author["group"]
            else:
                author_name = None
                author_group = None

            if not content or not isinstance(content, str):
                content = ""
            if not summary or not isinstance(summary, str):
                summary = ""

            content = bs4.BeautifulSoup(content, "html.parser").get_text(" ")
            summary = bs4.BeautifulSoup(summary, "html.parser").get_text(" ")

            writer.writerow(
                [
                    date,
                    debate,
                    dossier,
                    number,
                    author_name,
                    author_group,
                    content,
                    summary,
                ]
            )
