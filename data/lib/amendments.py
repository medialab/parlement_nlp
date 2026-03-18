from collections import defaultdict

import ebbe
import casanova

class Amendments:
    def __init__(self, csv_path):
        self.db = defaultdict(lambda: defaultdict(list))

        with casanova.reader(csv_path) as source:
            for row in source:
                date, debat, _, number, author_name, author_group, content, summary = row

                amendment = {
                    "debate": debat,
                    "number": number,
                    "author_name": author_name,
                    "author_group": author_group,
                    "amendment_content": content,
                    "amendment_summary": summary
                }

                #assert self.db[date][numero][texte].append(amendement.copy())

                self.db[date][number].append(amendment.copy())

    def get(self, date, number):
        return ebbe.getpath(self.db, [date, number], [])
