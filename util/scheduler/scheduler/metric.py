# Variety of scheduling metrics.

from __future__ import annotations
from typing import *

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
# Device/Kernel profile data structures.

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

    def __len__(self) -> int:
        return len(self.kernel_profiles[0])

    def __repr__(self) -> str:
        return f'DeviceProfile(device={self.device_name})'
