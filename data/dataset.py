import json
import casanova
import argparse

parser = argparse.ArgumentParser(description="Parse debate data into CSV")
parser.add_argument("--input", help="Path to input JSON file")
parser.add_argument("--output", help="Path to output CSV file")
args = parser.parse_args()

DEBATS_JSON = args.input
OUTPUT_PATH = args.output

HEADERS = [
    "speech_id",
    "debate_id",
    "amendment_author_name",
    "amendment_author_group",
    "amendment_content",
    "amendment_summary",
    "speech_date",
    "speaker_name",
    "speaker_group",
    "speaker_government",
    "label",
    "speech"
]

with open(DEBATS_JSON) as source:
    data = json.load(source)

with casanova.writer(OUTPUT_PATH, HEADERS) as writer:
    for debat in data:
        debate_id = debat["vote_id"]
        speech_date = debat["date"]
        amendment_author_name = debat["author_name"]
        amendment_author_group = debat["author_group"]
        amendment_content = debat["content"]
        amendment_summary = debat["summary"]

        subject = debat["subject"]

        for tour in debat["turns"]:
            speaker_name = tour["speaker"]["name"]
            speaker_party = tour["speaker"]["group_harmonized"]
            speaker_gouv = tour["speaker"]["government"]

            if not speaker_gouv and not speaker_party:
                continue

            if speaker_gouv and not speaker_party:
                speaker_party = "GOUV"

            label = tour["vote"]
            
            # On skip les abstention et l'absence de vote
            if not label: continue
            if label == "ABSTENTION": continue

            speech = ' '.join([parole["speech"] for parole in tour["speech"]])
            speech_id = tour["speech"][0]["speech_id"]
           

            writer.writerow([
                speech_id,
                debate_id,
                amendment_author_name,
                amendment_author_group,
                amendment_content,
                amendment_summary,
                speech_date,
                speaker_name,
                speaker_party,
                speaker_gouv,
                label,
                speech
            ])