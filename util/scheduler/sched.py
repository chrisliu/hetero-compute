import argparse
import os
import yaml

import scheduler

def main():
    parser = create_parser()
    args = parser.parse_args()
    print(args.profiles)

def create_parser() -> argparse.ArgumentParser:
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

if __name__ == '__main__':
    main()

