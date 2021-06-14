# Variety of scheduler implementations and data structures.

from __future__ import annotations
import copy
from typing import *

class KernelSegment:
    """POD that contains a kernel and a segment."""

    def __init__(self, kernel_name: str, segment: int, exec_time: float):
        self.kernel_name = kernel_name
        self.segment     = segment
        self.exec_time   = exec_time

class DeviceSchedule:
    """Schedule for a single device."""
    
    def __init__(self, device_name: str):
        self.device_name: str                 = device_name
        self.exec_time  : float               = float(0)
        self.schedule   : List[KernelSegment] = list()

class Scheduler:
    def schedule(self, config: Dict[str, int], metric: Metric) \
            -> Tuple[Metric, List[DeviceSchedule]]:
        raise NotImplemented("schedule not implemented.")

class ExhaustiveScheduler(Scheduler):
    def __init__(self, profiles: List[DeviceProfile]):
        self.profiles = profiles

    def schedule(self, config: Dict[str, int], metric: Metric) \
            -> Tuple[Metric, List[DeviceSchedule]]:
        """Returns the best schedule given a configuration of devices."""
        # Initialize devices.
        schedules: Dict[str, DeviceSchedule] = {
            device_name: [DeviceSchedule(device_name) for _ in range(count)]
            for (device_name, count) in config.items()
        }

        # Get optimal schedule.
        (max_time, schedules) = self.__schedule_impl(schedules, metric)
        
        # Return flattened schedule.
        flat_schedules = [sched for (_, sched_list) in schedules.items()
                         for sched in sched_list]
        return (max_time, flat_schedules) 

    def __schedule_impl(self, schedules: Dict[str, DeviceSchedule], 
                        metric: Metric, segment=0) \
            -> Tuple[Metric, Dict[str, DeviceSchedule]]:
        # Base case.
        # No more segments to reassign.
        if segment == len(self.profiles[0]):
            return (metric.compute_metric(schedules), schedules)
            #return (metric.compute_metric(schedules), copy.deepcopy(schedules))

        # Recursive case.
        best_profile = (metric.worst_metric(), dict())

        # For each device.
        for dev_profile in self.profiles:
            dev_name = dev_profile.device_name
            # For each kernel.
            for kernel_profile in dev_profile.kernel_profiles:
                kerseg = KernelSegment(kernel_profile.kernel_name, segment,
                                       kernel_profile[segment])
                # For each scheduled device.
                for dev_sched in schedules[dev_name]:
                    # What happensd if this segment is given to this 
                    # device with this kernel.
                    dev_sched.exec_time += kernel_profile[segment]
                    dev_sched.schedule.append(kerseg)
                    best_result = self.__schedule_impl(schedules, metric, 
                                                       segment + 1)
                    # Update profile if needed.
                    if best_result[0] < best_profile[0]:
                        if segment == len(self.profiles[0]) - 1:
                            (metric, sched) = best_result
                            best_profile = (metric, copy.deepcopy(sched))
                        else:
                            best_profile  = best_result
                        #best_profile  = best_result

                    # Restore previous schedule.
                    dev_sched.exec_time -= kernel_profile[segment]
                    dev_sched.schedule.pop()


        return best_profile

def pprint_schedule(schedule: List[DeviceSchedule]):
    num_devices     = len(schedule)
    max_device_name = 0
    max_kernel_name = 0
    max_segment     = 0
    max_exec_time   = 0
    max_device_id   = len(f'Device {num_devices - 1}')

    # Dict[segment id, Tuple(device id, kernel name, exec time)]
    segment_map = dict()
    # Dict[device id, Tuple(device name, exec time)]
    device_map  = dict()

    # Gather formatted data.
    for i, dev_sched in enumerate(schedule):
        # Update device name.
        max_device_name = max(max_device_name, len(dev_sched.device_name))
        max_exec_time   = max(max_exec_time, dev_sched.exec_time)
        device_map[i]   = (dev_sched.device_name, dev_sched.exec_time)

        for seg in dev_sched.schedule:
            max_kernel_name = max(max_kernel_name, len(seg.kernel_name))
            max_exec_time   = max(max_exec_time, seg.exec_time)
            max_segment     = max(max_segment, seg.segment)
            segment_map[seg.segment] = (i, seg.kernel_name, seg.exec_time)

    max_exec_time = len(f'[{max_exec_time:0.2f} ms]')
    max_main_col  = max(max_device_name, max_kernel_name, max_exec_time,
                        max_device_id) + 2
    max_seg_col   = max(len('Segment'), len(str(max_segment))) + 2

    # Row divider.
    row = '+' + '-' * max_seg_col + '+' + \
        ('-' * max_main_col + '+') * num_devices
    
    # Print device header.
    print(row)
    print(f"|{' ' * max_seg_col}|" + 
          ''.join(f'Device {i}'.center(max_main_col) + '|'
                  for i in range(num_devices)))
    print(f"|{' ' * max_seg_col}|" + 
          ''.join(device_map[i][0].center(max_main_col) + '|'
                  for i in range(num_devices)))
    print(f"|{'Segment'.center(max_seg_col)}|" + 
          ''.join(f'[{device_map[i][1]:0.2f} ms]'.center(max_main_col) + '|'
                  for i in range(num_devices)))
    print(row)

    # Print segment information.
    for seg in range(max_segment + 1):
        # First row.
        print("|" + str(seg).center(max_seg_col) + "|", end="")
        for dev in range(num_devices):
            if dev == segment_map[seg][0]:
                print(segment_map[seg][1].center(max_main_col) + "|", end="")
            else:
                print(" " * max_main_col + "|", end="")
        print()

        # Second row.
        print("|" + " " * max_seg_col + "|", end="")
        for dev in range(num_devices):
            if dev == segment_map[seg][0]:
                exec_time = segment_map[seg][2]
                print(f"[{exec_time:0.2f} ms]".center(max_main_col) + "|",
                      end="")
            else:
                print(" " * max_main_col + "|", end="")
        print()

        print(row)
