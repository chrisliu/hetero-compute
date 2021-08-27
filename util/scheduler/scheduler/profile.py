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

PushPullEpochSchedule = Tuple[List[DeviceProfile], List[DeviceProfile]]
PushPullSchedule      = List[PushPullEpochSchedule]
