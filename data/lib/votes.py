import json

from collections import Counter, defaultdict

class Votes:
    def __init__(self, json_path):
        if isinstance(json_path, str):
            with open(json_path) as source:
                self.db = json.load(source)
        else:
            self.db = json.load(json_path)

        # Scrutin par groupes
        for vote in self.db.values():
            votes = [(v["group"], v["vote"]) for v in vote["votes"]]
            vote["group"] = self._tuple_to_counter(votes)

    def _tuple_to_counter(self, tuples):
        counter = defaultdict(Counter)
        for a, b in tuples:
            counter[a][b] += 1

        vote_dict = {}
        for key, value in counter.items():
            vote_dict[key] = value.most_common(1)[0][0]

        return vote_dict

    def get(self, id):
        return self.db[id]

    def filter(self, condition, *args, **kwargs):
        for value in self.db.values():
            if condition(value, *args, **kwargs):
                yield value

    def from_name(self, id, name):
        for obj in self.db[id]["votes"]:
            if obj["name"] == name:
                return obj["vote"]

        return None

    def from_group(self, id, group):
        scrutin = self.db[id]
        return scrutin["group"].get(group)
