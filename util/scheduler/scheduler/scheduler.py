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

class KernelSegment:
    """POD that contains a kernel and a segment."""

    def __init__(self, kernel_name: str, segment: int):
        self.kernel_name = kernel_name
        self.segment     = segment

class DeviceSchedule:
    """Schedule for a single device."""
    
    def __init__(self, device_name: str):
        self.device_name                      = device_name
        self.exec_time  : float               = float(0)
        self.schedule   : List[KernelSegment] = list()

class Metric:
    def __lt__(self, other) -> bool:
        raise NotImplemented("metric comparison not implemented!")

    @staticmethod
    def compute_metric(schedules: Dict[str, DeviceSchedule]):
        raise NotImplemented("compute_metric not implemented.")

    @staticmethod
    def worst_metric():
        raise NotImplemented("compute_metric not implemented.")

class BestMaxTimeMetric(Metric):
    def __init__(self, worst_time: float, delta: float):
        self.worst_time = worst_time
        self.delta      = delta

    def __lt__(self, other) -> bool:
        # Prioritize worst time.
        # If equal, pick the one that is least balanced since it's using the
        # kernels for each device.
        return self.worst_time < other.worst_time or \
            (self.worst_time == other.worst_time and self.delta > other.delta)

    @staticmethod
    def compute_metric(schedules: Dict[str, DeviceSchedule]):
        exec_times = [sched.exec_time
                      for (_, sched_list) in schedules.items()
                      for sched in sched_list]
        worst_time = max(exec_times)
        delta      = sum((worst_time - exec_time) ** 2
                         for exec_time in exec_times)
        return BestMaxTimeMetric(worst_time, delta)

    @staticmethod
    def worst_metric():
        return BestMaxTimeMetric(float('inf'), float('inf'))

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
                kerseg = KernelSegment(kernel_profile.kernel_name, segment)
                # For each schedueld device.
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

                    # Update porfile if needed.
                    if best_result[0] < best_profile[0]:
                        best_profile  = best_result

        return best_profile

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

    print(f"Expected time: {schedule[0].worst_time}")
    print("Schedule:")
    for dev_sched in schedule[1]:
        print(f" > {dev_sched.device_name} executes in {dev_sched.exec_time}")
        print("    > ", end="")
        for seg in dev_sched.schedule:
            print(f"{seg.segment}:{seg.kernel_name} ", end="")
        print()

