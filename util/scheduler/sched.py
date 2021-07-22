import argparse
import os
import time
import yaml

import scheduler

from collections import defaultdict
from typing import *

def main():
    # Load arguments.
    parser = create_parser()
    args = parser.parse_args()

    # Load profiles and query device counts.
    profiles = load_profiles(args.profiles)
    #hardware_config = query_devices(profiles)
    #hardware_config = {'Intel Xeon E5-2686': 1, 'NVIDIA Tesla M60': 2}
    #hardware_config = {'Intel i7-9700K': 1, 'NVIDIA Quadro RTX 4000': 2}
    #hardware_config = {'Intel i7-9700K': 1, 'NVIDIA Quadro RTX 4000': 1}
    #hardware_config = {'NVIDIA Quadro RTX 4000': 2}
    hardware_config = {'NVIDIA Quadro RTX 4000': 1}
    profiles = filter_profiles(profiles, hardware_config)

    # Schedule.
    s = scheduler.MostGainScheduler(profiles)

    start_t = time.time()
    schedule = s.schedule(hardware_config)
    end_t = time.time()
    print(f"Scheduler took {end_t - start_t:0.2f} seconds.")

    # Display schedule.
    scheduler.pprint_schedule(schedule)

    # Print speedup against single device.
    worst_time = max(schedule, key=lambda sched: sched.exec_time).exec_time
    single_dev_time = s.best_single_device_time()
    print(f"Longest device time:     {worst_time:0.2f} ms")
    print(f"Best single device time: {single_dev_time:0.2f} ms")
    print(f"{single_dev_time / worst_time:0.2f}x speedup")

    # Save schedule.
    scheduler.contiguify_schedule(schedule)
    write_schedule(schedule, 'out.skd')

    # Write out SSSP hetero file.
    with open('sssp_pull.cuh', 'w') as ofs:
        ofs.write(
            scheduler.kernelgen.generate_sssp_hetero_source_code(schedule))

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

def query_devices(profiles: List[scheduler.DeviceProfile]) -> Dict[str, int]:
    """Query user for devices."""
    def valid_count(devname: str) -> int:
        """Returns first valid integer value inputted."""
        print(f"How many {devname}?")
        while True:
            val = input(" >  ")
            try:
                return int(val)
            except:
                print("(!) Please enter an integer.")

    hardware_config = dict()
    for devprof in profiles:
        devname = devprof.device_name
        count   = valid_count(devname)
        if count: # Only add device if count > 0.
            hardware_config[devname] = count
    return hardware_config

def filter_profiles(profiles: List[scheduler.DeviceProfile], 
                    hardware_config: Dict[str, int]) \
        -> List[scheduler.DeviceProfile]:
    """Removes device profiles that aren't part of the hardware configuration.
    """
    return [devprof for devprof in profiles
            if devprof.device_name in hardware_config]

def write_schedule(schedule: List[scheduler.DeviceSchedule], fname: str):
    """Writes schedule out to file.

    Sample output:
    Procesor 1
    0 1
    Kernel 1
    3 3
    Kernel 2
    Processor 2
    2 2
    Kernel 3
    4 5
    Kernel 4
    """
    with open(fname, 'w') as ofs:
        for devsched in schedule:
            # Write filename.
            ofs.write(f'{devsched.device_name}\n')

            # Write segments and kernels.
            for seg in devsched.schedule:
                ofs.write(f'{seg.seg_start} {seg.seg_end}\n')
                ofs.write(f'{seg.kernel_name}\n')

if __name__ == '__main__':
    main()
