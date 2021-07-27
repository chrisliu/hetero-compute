/**
 * Device type declarations.
 */

#ifndef SRC__DEVICES_H
#define SRC__DEVICES_H

#include <ostream>
#include <string>

/*****************************************************************************
 ***** Identifiers ***********************************************************
 *****************************************************************************/

/** Available devices. */
enum class Device {
    intel_i7_9700K, intel_xeon_e5_2686,
    nvidia_quadro_rtx_4000, nvidia_tesla_k80, nvidia_tesla_m60,
    undefined
};

// List of devices. No elegant way of iterating through enums.
constexpr Device DeviceList[] = { 
    Device::intel_i7_9700K, 
    Device::intel_xeon_e5_2686,
    Device::nvidia_quadro_rtx_4000,
    Device::nvidia_tesla_k80,
    Device::nvidia_tesla_m60,
    Device::undefined
};

/** Types of devices. */
enum class DeviceType {
    CPU, GPU, undefined
};

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

/**
 * Converts device ID to its name.
 * Parameters:
 *   - dev <- device ID.
 * Returns:
 *  device name.
 */
std::string to_string(Device dev) {
    switch (dev) {
        case Device::intel_i7_9700K:         return "Intel i7-9700K";
        case Device::intel_xeon_e5_2686:     return "Intel Xeon E5-2686";
        case Device::nvidia_quadro_rtx_4000: return "NVIDIA Quadro RTX 4000";
        case Device::nvidia_tesla_k80:       return "NVIDIA Tesla K80";
        case Device::nvidia_tesla_m60:       return "NVIDIA Tesla M60";
        default:                             return "undefined device";
    }
}

/**
 * Get Device from device string name. 
 * Parameters:
 *   - devstr <- string name of device.
 * Returns:
 *   Corresponding device. If no match, Device::undefined.
 */
Device get_device(std::string &devstr) {
    for (Device dev : DeviceList)
        if (devstr == to_string(dev)) return dev;

    return Device::undefined;
}

/**
 * Get device type.
 * Parameters:
 *   - dev <- device ID.
 * Returns:
 *   Device type.
 */
DeviceType get_device_type(Device dev) {
    switch(dev) {
        case Device::intel_i7_9700K:         
        case Device::intel_xeon_e5_2686:     return DeviceType::CPU;
        case Device::nvidia_quadro_rtx_4000:
        case Device::nvidia_tesla_k80:
        case Device::nvidia_tesla_m60:       return DeviceType::GPU;
        case Device::undefined:
        default:                             return DeviceType::undefined;
    }
}

/*****************************************************************************
 ***** I/O Functions *********************************************************
 *****************************************************************************/

std::ostream &operator<<(std::ostream &os, Device dev) {
    os << to_string(dev);
    return os;
}

std::ostream &operator<<(std::ostream &os, DeviceType devt) {
    switch(devt) {
        case DeviceType::CPU:       os << "CPU"; break;
        case DeviceType::GPU:       os << "GPU"; break;
        case DeviceType::undefined: os << "undefined device type"; break;
    }
    return os;
}

#endif // SRC__DEVICES_H
