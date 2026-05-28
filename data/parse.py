import casanova
import argparse
import regex
import sys

from nltk.tokenize import PunktSentenceTokenizer


def err(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def listgen(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return list(result)

    return wrapper


HEADER = [
    "sub_sub_subject",
    "amendments",
    "amendments_precise",
]

PRONOMS = ["je", "ils", "elles", "il", "elle", "vous", "nous"]

AMENDEMENT = regex.compile(
    r"(?:(?:sous-)?amendements?\s*(?:identiques?)?)?\s*n\s*os?(?:\s| )*(?:s\s* )?\s*(?:(?:(?:,?)|(?:\s*et))?(?:(?:\s*(?P<number>[0-9]+)(?:\sr[ée]ctifi[ée]s?)?)|(?:\s*(?P<number>suivants?))))+",
    flags=regex.M | regex.UNICODE,
)

RAPPEL_REGLEMENT = regex.compile(r"[Rr]appel.+?[Rr][éèe]glement")

SENTENCE = regex.compile(r".*(\.|\?|!|…)$")

ACTIONS = (
    ("appel_ordre_jour", regex.compile(r"(ordre du jour appelle)", flags=regex.I)),
    ("demande_scrutin", regex.compile(r"(demandes? de scrutins?)", flags=regex.I)),
    ("saisie_amendements", regex.compile(r"(suis saisi)", flags=regex.I)),
    ("soutenir_amendement", regex.compile(r"(pour soutenir)", flags=regex.I)),
    ("mise_aux_voix", regex.compile(r"(mets aux voix)", flags=regex.I)),
    (
        "don_parole",
        regex.compile(
            r"(avez la parole)|(parole est à)|((conservez|gardez) la parole)",
            flags=regex.I,
        ),
    ),
    ("discussion_commune", regex.compile(r"(discussion commune)", flags=regex.I)),
    (
        "objet_sous_amendement",
        regex.compile(r"(objet.+?sous-amendement)", flags=regex.I),
    ),
    (
        "defendu",
        regex.compile(
            r"(sont (pas )?d[ée]fendus)|(est (pas )?d[ée]fendu)", flags=regex.I
        ),
    ),
)

punkt = PunktSentenceTokenizer()


def get_action(string):
    # ==== parse des actions de la présidence ====

    # avis du gouvernement (aussi : « pour donner l'avis »)
    # avis de la commission
    # saisie d'une demande de scrutins public
    # la parole est à (...) soutenir l'amendement | parole est à (...) pour les soutenir
    # met au voix l'amendement
    # (...) a été défendu
    # faire l'objet d'une présentation groupée
    #

    for label, expr in ACTIONS:
        if expr.search(string):
            return label

    return None


def is_title(string):
    if SENTENCE.match(string.strip()):
        return False

    if len(string) > 100:
        return False

    if any(a in string.lower() for a in PRONOMS):
        return False

    return True


@listgen
def get_amendments(string):
    for m in AMENDEMENT.finditer(string):
        for n in m.captures("number"):
            yield n


def parse_moderation(row, state):
    _, intervention, function, sub_subject, code = (
        row[0],
        row[13],
        row[11],
        row[6],
        row[12],
    )
    in_discussion, direct_amendment, last_sub_subject, last_sub_sub_subject = state

    if code != "PAROLE_GENERIQUE":
        return state

    if last_sub_subject != sub_subject and not RAPPEL_REGLEMENT.search(sub_subject):
        in_discussion, direct_amendment, last_sub_subject, last_sub_sub_subject = (
            set(),
            set(),
            sub_subject,
            last_sub_sub_subject,
        )

    if function != "président":
        return in_discussion, direct_amendment, last_sub_subject, last_sub_sub_subject

    if is_title(intervention):
        if RAPPEL_REGLEMENT.search(intervention) or RAPPEL_REGLEMENT.search(
            str(last_sub_sub_subject)
        ):
            return in_discussion, direct_amendment, last_sub_subject, intervention
        else:
            return set(), set(), last_sub_subject, intervention

    sentences = punkt.tokenize(intervention)

    for sentence in sentences:
        action = get_action(sentence)
        amendments = set(get_amendments(sentence))

        match action:
            case "demande_scrutin":
                continue
            case "saisie_amendements":
                in_discussion = amendments if amendments else in_discussion
                direct_amendment = set()
            case "soutenir_amendement":
                if len(in_discussion.intersection(amendments)) > 0:
                    direct_amendment = amendments
                else:
                    direct_amendment = amendments
                    in_discussion = amendments
            case "discussion_commune":
                in_discussion = amendments if amendments else in_discussion
                direct_amendment = set()
            case "objet_sous_amendement":
                in_discussion.update(amendments)
                direct_amendment = set()
            case "defendu":
                direct_amendment = set()
            case _:
                direct_amendment = set()

    return in_discussion, direct_amendment, last_sub_subject, last_sub_sub_subject


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file")
    parser.add_argument("input", help="Input filename path")
    args = parser.parse_args()

    state = set(), set(), None, None
    with open(args.input) as source:
        enricher = casanova.enricher(source, sys.stdout, add=HEADER)
        for row in enricher:
            state = parse_moderation(row, state)
            in_discussion, direct_amendment, last_subject, last_sub_sub_subject = state
            enricher.writerow(
                row,
                [
                    last_sub_sub_subject,
                    "|".join(in_discussion),
                    "|".join(direct_amendment),
                ],
            )
