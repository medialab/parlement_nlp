import pandas
import argparse

SEED = 42

parser = argparse.ArgumentParser(description="Process French parliament debate and amendment data for BERT fine-tuning.")
parser.add_argument("--csv-15th-debates", help="Path to 15th debates CSV")
parser.add_argument("--csv-16th-debates",  help="Path to 16th debates CSV")
parser.add_argument("--csv-17th-debates",  help="Path to 17th debates CSV")
parser.add_argument("--csv-15th-amendments", help="Path to 15th amendments CSV")
parser.add_argument("--csv-16th-amendments", help="Path to 16th amendments CSV")
parser.add_argument("--csv-17th-amendments", help="Path to 17th amendments CSV")
parser.add_argument("--output", default="./export/fine-tuning-bert.csv", help="Output CSV path")
args = parser.parse_args()

df_15th_debates = pandas.read_csv(args.csv_15th_debates)
df_16th_debates = pandas.read_csv(args.csv_16th_debates)
df_17th_debates = pandas.read_csv(args.csv_17th_debates)

df_debates = pandas.concat([df_15th_debates, df_16th_debates, df_17th_debates])

df_debates = df_debates[df_debates["code"] == "PAROLE_GENERIQUE"]
df_debates = df_debates[df_debates["function"] != "président"]

df_debates_grps = (df_debates['name'] != df_debates['name'].shift()).cumsum()
df_debates = (df_debates.groupby(["name", df_debates_grps], sort=False)["intervention"]
              .agg(' '.join)
              .reset_index(level=1, drop=True)
              .reset_index())

df_debates = df_debates[["intervention"]]
df_debates = df_debates.drop_duplicates()
df_debates = df_debates.rename(columns={"intervention": "text"})
df_debates = df_debates.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_debates = df_debates[:33325]

df_15th_amendments = pandas.read_csv(args.csv_15th_amendments)
df_16th_amendments = pandas.read_csv(args.csv_16th_amendments)
df_17th_amendments = pandas.read_csv(args.csv_17th_amendments)

df_amendments = pandas.concat([df_15th_amendments, df_16th_amendments, df_17th_amendments])

df_amendments = df_amendments[["amendment_summary"]]
df_amendments = df_amendments.rename(columns={"amendment_summary": "text"})
df_amendments = df_amendments.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_amendments = df_amendments[:1150]

df = pandas.concat([df_debates, df_amendments])
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv(args.output, index=False)