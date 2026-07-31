import pandas as pd
import argparse

def yes_no(txt):
    print(txt)
    c = input("Keep ?")
    print("-" * 200)
    if 'n' in c.lower():
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV path")
    parser.add_argument("column", help="Column to read")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["keep"] = df[args.column].apply(yes_no)
    
    df = df[df["keep"]]

    df.to_csv(args.output, index=None)

