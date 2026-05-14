#!/usr/bin/env python3

import argparse
import csv
import random
import re
import subprocess
from pathlib import Path

EXPERIMENTS = [
    ("random", "experiments/Assignment_Random.xml"),
    ("clustered", "experiments/Assignment_Clustered.xml"),
    ("powerlaw", "experiments/Assignment_Powerlaw.xml"),
]

SCORE_PATTERN = re.compile(
    r"^\s*(?P<score>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"(?P<seed>[0-9]+)\s*$"
)

def make_temp_xml(original_xml, seed):
    original_path = Path(original_xml)
    text = original_path.read_text()

    text = re.sub(
        r'random_seed="[^"]*"',
        f'random_seed="{seed}"',
        text,
        count=1
    )

    temp_path = original_path.with_name(f"{original_path.stem}_seed_{seed}.xml")
    temp_path.write_text(text)

    return temp_path

def run_argos(xml_file):
    cmd = ["argos3", "-n", "-c", str(xml_file)]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if completed.returncode != 0:
        print("\nARGoS failed.")
        print("Command:")
        print(" ".join(cmd))
        print("\nARGoS output:")
        print(completed.stdout)
        raise RuntimeError("ARGoS run failed. See ARGoS output above.")

    return completed.stdout

def parse_score(output):
    for line in reversed(output.splitlines()):
        match = SCORE_PATTERN.match(line)
        if match:
            return float(match.group("score"))

    print("\nCould not parse score.")
    print("Last 30 ARGoS output lines:")
    print("\n".join(output.splitlines()[-30:]))
    raise ValueError("Final score line not found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True) 
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--out", default="results/assignment_scores.csv")
    parser.add_argument("--keep-temp-xml", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()

    with output_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(["version", "distribution", "run", "seed", "collected_resources"])

        for distribution, xml_file in EXPERIMENTS:
            for run in range(1, args.runs + 1):
                seed = random.randint(1, 1000000)

                print(f"{args.version} | {distribution} | run {run}/{args.runs} | seed {seed}")

                temp_xml = make_temp_xml(xml_file, seed)

                try:
                    output = run_argos(temp_xml)
                    score = parse_score(output)

                    writer.writerow([args.version, distribution, run, seed, score])
                    csvfile.flush()

                finally:
                    if not args.keep_temp_xml and temp_xml.exists():
                        temp_xml.unlink()

if __name__ == "__main__":
    main()