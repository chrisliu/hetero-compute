from __future__ import annotations
import copy
from typing import *

class KernelProfile:
    """POD that contains the kernel's name and its respective profile."""

    def __init__(self, kernel_name, profile: List[float]):
        self.kernel_name = kernel_name
        self.__profile   = profile

    def __getitem__(self, key: int) -> float:
        """profile getter sugar."""
        return self.__profile[key]

    def __len__(self) -> int:
        return len(self.__profile)

    def __repr__(self) -> str:
        return f'KernelProfile(kernel={self.kernel_name})'

class DeviceProfile:
    """POD that contains the execution profiles for various kernels on a single
    device."""

    def __init__(self, device_name: str, kernel_profiles: List[KernelProfile]):
        self.device_name     = device_name
        self.kernel_profiles = kernel_profiles

    def is_device(self, dname: str) -> bool:
        """Returns if this device profile corresponds to the requested device."""
        return dname == self.device_name

    def __len__(self) -> int:
        return len(self.kernel_profiles[0])

    def __repr__(self) -> str:
        return f'DeviceProfile(device={self.device_name})'

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

class Metric:
    def __lt__(self, other: Metric) -> bool:
        raise NotImplemented("metric comparison not implemented!")

    @staticmethod
    def compute_metric(schedules: Dict[str, DeviceSchedule]) -> Metric:
        raise NotImplemented("compute_metric not implemented.")

    @staticmethod
    def worst_metric() -> Metric:
        raise NotImplemented("compute_metric not implemented.")

class BestMaxTimeMetric(Metric):
    """
    Metric that ranks by:
      1) Best worst-possible device time.
      2) Min overall time.
      3) Least least-squares (from best worst-possible device time).
    """
    def __init__(self, worst_time: float, overall_time: float, variance: float):
        self.worst_time   = worst_time
        self.overall_time = overall_time
        self.variance     = variance

    def __lt__(self, other: BestMaxTimeMetric) -> bool:
        # If best worst-possible device time is better.
        if self.worst_time < other.worst_time:
            return True
        # If best worst-possible time is tied.
        elif self.worst_time == other.worst_time:
            # If overall time is better.
            if self.overall_time < other.overall_time:
                return True
            #If overall time is tied.
            elif self.overall_time == other.overall_time:
                # If variance is better.
                if self.variance < other.variance:
                    return True
        return False

    @staticmethod
    def compute_metric(schedules: Dict[str, DeviceSchedule]) \
            -> BestMaxTimeMetric:
        exec_times = [sched.exec_time
                      for (_, sched_list) in schedules.items()
                      for sched in sched_list]
        worst_time   = max(exec_times)
        overall_time = sum(exec_times)
        variance     = sum((worst_time - exec_time) ** 2
                           for exec_time in exec_times)
        return BestMaxTimeMetric(worst_time, overall_time, variance)

    @staticmethod
    def worst_metric() -> BestMaxTimeMetric:
        return BestMaxTimeMetric(float('inf'), float('inf'), float('inf'))

class Scheduler:
    def __init__(self, profiles: List[DeviceProfile]):
        self.profiles = profiles

    def schedule(self, config: Dict[str, int], metric: Metric) \
            -> Tuple[float, List[DeviceSchedule]]:
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
            return (metric.compute_metric(schedules), copy.deepcopy(schedules))

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
                    # Restore previous schedule.
                    dev_sched.exec_time -= kernel_profile[segment]
                    dev_sched.schedule.pop()

                    # Update profile if needed.
                    if best_result[0] < best_profile[0]:
                        best_profile  = best_result

        return best_profile

def pprint_schedule(schedule: List[DeviceSchedule]):
    print_out = """
-------------------------------------------------------------
|         |        Device 0        |       Device 1         |
|         |     Intel i7-9700K     | NVIDIA Quadro RTX 4000 |
| Segment |       [60.00 ms]       |      [55.00 ms]        |
-------------------------------------------------------------
|    9    |                        |     GPU_Kernel_1       |
|         |                        |      [35.00 ms]        |
-------------------------------------------------------------
|   10    |      CPU_Kernel_2      |                        |
|         |       [37.00 ms]       |                        |
-------------------------------------------------------------
|   11    |                        |     GPU_Kernel_2       |
|         |                        |      [20.00 ms]        |
-------------------------------------------------------------
|   12    |      CPU_Kernel_2      |                        |
|         |       [23.00 ms]       |                        |
-------------------------------------------------------------
    """
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
    row = '-' * (1 + max_seg_col + 1 + (max_main_col + 1) * num_devices)
    
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

if __name__ == '__main__':
    cpu_k1 = KernelProfile("CPU_Kernel_1", [100, 75, 50, 40])
    cpu_k2 = KernelProfile("CPU_Kernel_2", [120, 60, 55, 35])
    gpu_k1 = KernelProfile("GPU_Kernel_1", [ 40, 70 ,60, 50])
    gpu_k2 = KernelProfile("GPU_Kernel_2", [ 80, 55, 40, 20])

    cpu_profile = DeviceProfile("CPU", [cpu_k1, cpu_k2])
    gpu_profile = DeviceProfile("GPU", [gpu_k1, gpu_k2])
    profiles    = [cpu_profile, gpu_profile]

    hardware_config = {"CPU": 1, "GPU": 2}

    scheduler = Scheduler(profiles)
    schedule = scheduler.schedule(hardware_config, BestMaxTimeMetric)
    pprint_schedule(schedule[1])
