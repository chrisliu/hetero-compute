import argparse
import os
import yaml

import scheduler

from typing import *

def main():
    parser = create_parser()
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    scheduler.Scheduler(profiles)

def create_parser() -> argparse.ArgumentParser:
    """Returns a valid python argument parser."""
    parser = argparse.ArgumentParser(description="Simulate and generate " \
                                     "a heterogeneous configuration of " \
                                     "kernels and graph segments.")

    ## Helper functions.
    def valid_yaml_file(fname: str) -> str:
        ext = os.path.splitext(fname)[1][1:]
        if ext != 'yaml':
            parser.error(f"benchmark profile \"{fname}\" must be a YAML file") 
        return fname

    ## Parser arguments.
    parser.add_argument('profiles', type=valid_yaml_file, nargs='+',
                        help="YAML benchmark profiles for devices")
    return parser

def load_profiles(fnames: List[str]) -> List[dict]:
    """Returns contents of profile files."""
    profiles = list()
    for fname in fnames:
        with open(fname, 'r') as ifs:
            profiles.append(yaml.full_load(ifs))
    return profiles

if __name__ == '__main__':
    main()

