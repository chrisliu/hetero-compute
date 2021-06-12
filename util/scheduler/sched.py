import argparse
import os
import yaml

import scheduler

from collections import defaultdict
from typing import *

def main():
    parser = create_parser()
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    s = scheduler.Scheduler(profiles)

    hardware_config = {"Intel i7-9700K": 1, "NVIDIA Quadro RTX 4000": 2}

    (metric, sched) = s.schedule(hardware_config, 
                                         scheduler.BestMaxTimeMetric)
    scheduler.pprint_schedule(sched)

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

def load_profiles(fnames: List[str]) -> List[scheduler.DeviceProfile]:
    """Returns contents of profile files."""
    device_map = defaultdict(list)
    for fname in fnames:
        with open(fname, 'r') as ifs:
            config_results = yaml.full_load(ifs)

        # Save segment execution times as a kernel profile.
        device_name    = config_results['config']['device']
        kernel_name    = config_results['config']['kernel']
        kernel_results = config_results['results'][0]['segments']
        kernel_execs   = [segment['millis'] for segment in kernel_results]
        kernel_profile = scheduler.KernelProfile(kernel_name, kernel_execs)

        device_map[device_name].append(kernel_profile)

    # Package Dict["device name", List[KernelProfile]] into List[DeviceProfile]
    return [
        scheduler.DeviceProfile(device_name, kernel_profiles)
        for device_name, kernel_profiles in device_map.items()
    ]

if __name__ == '__main__':
    main()

