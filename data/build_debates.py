from lib.debates import DebateBuilder
import argparse

parser = argparse.ArgumentParser(description="Build seances CSV file")
parser.add_argument("--dir-debates", help="Path to seances directory")
parser.add_argument("--output-csv", help="Path to output CSV file")

args = parser.parse_args()

DIR_SEANCES = args.dir_debates
OUTPUT_CSV = args.output_csv

builder = DebateBuilder(DIR_SEANCES, OUTPUT_CSV, format="csv")

