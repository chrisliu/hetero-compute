import argparse
import os
import time
import yaml

import scheduler

from collections import defaultdict
from typing import *

def main():
    parser = create_parser()
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    #s = scheduler.ExhaustiveScheduler(profiles)
    s = scheduler.MostGainScheduler(profiles)

    hardware_config = {"Intel i7-9700K": 1, "NVIDIA Quadro RTX 4000": 1}
    hardware_config = {"Intel i7-9700K": 1, "NVIDIA Quadro RTX 4000": 2}
    hardware_config = {"Intel i7-9700K": 1, "NVIDIA Quadro RTX 4000": 3}

    start_t = time.time()
    schedule = s.schedule(hardware_config)
    #schedule = s.schedule(hardware_config, scheduler.BestMaxTimeMetric)
    end_t = time.time()
    print(f"Scheduler took {end_t - start_t:0.2f} seconds.")

    # Disply schedule.
    scheduler.pprint_schedule(schedule)

    # Print speedup against single device.
    worst_time = max(schedule, key=lambda sched: sched.exec_time).exec_time
    single_dev_time = s.best_single_device_time()
    print(f"Longest device time:     {worst_time:0.2f} ms")
    print(f"Best single device time: {single_dev_time:0.2f} ms")
    print(f"{single_dev_time / worst_time:0.2f}x speedup")

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
