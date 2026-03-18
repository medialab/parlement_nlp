import glob
import json
import ebbe
import bs4
import casanova
import tqdm
import os

from os.path import join

from lib.actors import Actors
import argparse

parser = argparse.ArgumentParser(description="Build amendments CSV from JSON files.")
parser.add_argument("--amendments-dir", help="Directory containing amendments JSON files")
parser.add_argument("--output-csv", help="Output CSV file path")
args = parser.parse_args()

AMENDEMENTS_DIR = args.amendments_dir
OUTPUT_CSV = args.output_csv

actors = Actors()

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

if __name__ == "__main__":
    paths = list(glob.glob(join(AMENDEMENTS_DIR, "*/*/*.json")))

    with casanova.writer(OUTPUT_CSV, ["date", "debate", "bill", "number", "author_name", "author_group", "amendment_content", "amendment_summary"]) as writer:
        for path in tqdm.tqdm(paths):
            with open(path) as source:
                data = json.load(source)

            # text
            bill = ebbe.getpath(data, ["amendement", "texteLegislatifRef"])
            if not bill or type(bill) != str: continue

            # debate 
            debate = ebbe.getpath(data, ["amendement", "seanceDiscussionRef"])
            if not debate or type(debate) != str: continue
            
            # number
            number = ebbe.getpath(data, ["amendement", "identification", "numeroLong"])
            if not number or type(number) != str: continue

            # date sort
            date = ebbe.getpath(data, ["amendement", "cycleDeVie", "dateSort"])
            if not date or type(date) != str: continue
            date = date[:10]

            # dispositif
            content = ebbe.getpath(data, ["amendement", "corps", "contenuAuteur", "dispositif"])
            summary = ebbe.getpath(data, ["amendement", "corps", "contenuAuteur", "exposeSommaire"])

            # depute
            author = ebbe.getpath(data, ["amendement", "signataires", "auteur", "acteurRef"])
            if not author or type(author) != str: continue
            author = actors.get(author, date)
            
            if author:
                author_name = author["name"]
                author_group = author["group"]
            else:
                author_name = None
                author_group = None

            if not content or type(content) != str: 
                content = ""
            if not summary or type(summary) != str: 
                summary = ""

            content = bs4.BeautifulSoup(content, "html.parser").get_text(" ")
            summary = bs4.BeautifulSoup(summary, "html.parser").get_text(" ")

            writer.writerow([
                date, debate, bill, number, author_name, author_group, content, summary
            ])