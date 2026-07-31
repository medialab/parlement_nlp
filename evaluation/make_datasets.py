import argparse
import pandas as pd

from os.path import join
from os import makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        help="Output directory for the test datasets"
    )
    args = parser.parse_args()

    makedirs("datasets", exist_ok=True)

    sts = pd.read_parquet("hf://datasets/CATIE-AQ/frenchSTS/data/test-00000-of-00001.parquet")
    sts.to_csv(join(args.output, "test_sts.csv"), index=None)

    sick = pd.read_csv("hf://datasets/Lajavaness/SICK-fr/sick_test_fr.csv")
    sick.to_csv(join(args.output, "test_sick.csv"), index=None)