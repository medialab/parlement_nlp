import pandas as pd
import argparse

def yes_no(txt):
    print(txt)
    c = input("In favor ?")
    print("-" * 200)
    if 'n' in c.lower():
        return 0.0
    return 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV path")
    parser.add_argument("column", help="Column to read")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["score"] = df[args.column].apply(yes_no)
        

    df.to_csv(args.output, index=None)

