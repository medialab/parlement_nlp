import json
import sys
import casanova
import bs4

GROUPS = {
    "PO707869": "LR",
    "PO656010": "UDI",
    "": "",
    "PO645633": "NI",
    "PO656018": "GDR",
    "PO656022": "RRDP",
    "PO656014": "ECOLOS",
    "PO656002": "SOC",
}

if __name__ == "__main__":
    data = json.load(sys.stdin)

    amendements = data["textesEtAmendements"]["texteleg"][680]["amendements"][
        "amendement"
    ]

    with casanova.writer(
        sys.stdout, ["author_group", "amendement_content", "amendment_summary"]
    ) as writer:
        for amd in amendements:
            group = amd["signataires"]["auteur"]["groupePolitiqueRef"]
            dispositif = amd["corps"]["dispositif"]
            expose = amd["corps"]["exposeSommaire"]

            dispositif = bs4.BeautifulSoup(dispositif, "html.parser").get_text()
            expose = bs4.BeautifulSoup(expose, "html.parser").get_text()

            writer.writerow([group, dispositif, expose])
