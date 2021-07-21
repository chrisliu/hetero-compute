/**
 * Schedule file loading and parsing.
 */

#ifndef SRC__SCHEDULER_H
#define SRC__SCHEDULER_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "devices.h"
#include "graph.h"
#include "kernels/cpu/sssp_pull.h"
#include "kernels/gpu/sssp_pull.cuh"
#include "kernels/kernel_types.h"

/*****************************************************************************
 ***** Data Structures *******************************************************
 *****************************************************************************/

/** Kernel schedule block base type. */
struct ScheduleBlock {
    const int   device_id;     // Which device that this kernel belongs to.
    const nid_t start_segment; // Start segment (inclusive).
    const nid_t end_segment;   // End segment (inclusive).
    nid_t       start_id;      // Mutable start id (bookkeeping for kernel).
    nid_t       end_id;        // Mutable end id (bookkeeping for kernel).

    ScheduleBlock(int device_id_, nid_t start_segment_, nid_t end_segment_)
        : device_id(device_id_)
        , start_segment(start_segment_)
        , end_segment(end_segment_)
        , start_id(0)
        , end_id(0)
    {}
};

/** SSSP CPU schedule block type. */
struct SSSPCPUScheduleBlock : public ScheduleBlock {
    const sssp_cpu_epoch_func kernel;      // SSSP CPU kernel.
    const SSSPCPU             kernel_type; // Type of kernel for helper funcs.

    SSSPCPUScheduleBlock(int device_id_, nid_t start_segment_, 
            nid_t end_segment_, sssp_cpu_epoch_func kernel_,
            SSSPCPU kernel_type_)
        : ScheduleBlock(device_id_, start_segment_, end_segment_)
        , kernel(kernel_)
        , kernel_type(kernel_type_)
    {}
};

/** SSSP GPU schedule block type. */
struct SSSPGPUScheduleBlock : public ScheduleBlock {
    const sssp_gpu_epoch_func kernel;       // SSSP GPU kernel.
    const SSSPGPU             kernel_type;  // Type of kernel for helper funcs.
    const int                 block_count;  // Block count.
    const int                 thread_count; // Thread count.

    SSSPGPUScheduleBlock(int device_id_, nid_t start_segment_, 
            nid_t end_segment_, sssp_gpu_epoch_func kernel_,
            SSSPGPU kernel_type_, int block_count_, int thread_count_)
        : ScheduleBlock(device_id_, start_segment_, end_segment_)
        , kernel(kernel_)
        , kernel_type(kernel_type_)
        , block_count(block_count_) 
        , thread_count(thread_count_)
    {}
};

/** Container for heterogeneous SSSP schedule. */
using SSSPScheduleBlock = std::variant<SSSPCPUScheduleBlock, 
                                       SSSPGPUScheduleBlock>;
struct SSSPHeteroSchedule {
    std::vector<SSSPScheduleBlock> blocks; // Blocks in this schedule.
    nid_t num_segments;                    // Total number of segments 
                                           // (not blocks).
};

/** SSSP CPU block emitter. */
std::ostream &operator<<(std::ostream &os, SSSPCPUScheduleBlock &block) {
    os << "SSSP CPU Block" << std::endl
       << " > device id:     " << block.device_id << std::endl
       << " > start segment: " << block.start_segment << std::endl
       << " > end segment:   " << block.end_segment << std::endl
       << " > kernel:        " << block.kernel_type << std::endl;
    return os;
}

/** SSSP GPU block emitter. */
std::ostream &operator<<(std::ostream &os, SSSPGPUScheduleBlock &block) {
    os << "SSSP GPU Block" << std::endl
       << " > device id:     " << block.device_id << std::endl
       << " > start segment: " << block.start_segment << std::endl
       << " > end segment:   " << block.end_segment << std::endl
       << " > kernel:        " << block.kernel_type << std::endl
       << " > block count:   " << block.block_count << std::endl
       << " > thread count:  " << block.thread_count << std::endl;
    return os;
}

/** SSSP heterogeneous schedule emitter. */
std::ostream &operator<<(std::ostream &os, SSSPHeteroSchedule &schedule) {
    os << "SSSP Heterogeneous Schedule" << std::endl
       << " > Number of segments: " << schedule.num_segments << std::endl
       << "Blocks ..." << std::endl;
    
    for (auto block : schedule.blocks)
        std::visit([&os](auto && arg){ os << arg; }, block);
    return os;
}

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

/**
 * Get kernel from string.
 * Parameters:
 *   - kerstr <- string form of kernel (including opt args).
 * Returns:
 *   Kernel enum corresponding to the kernel.
 */
template <typename EnumT, 
         typename = std::enable_if_t<std::is_enum<EnumT>::value>>
EnumT get_kernel_id(std::string kerstr) {
    for (EnumT ker : get_kernels(EnumT::undefined)) {
        std::string cand_kerstr = to_string(ker);
        if (kerstr.substr(0, cand_kerstr.length()) == cand_kerstr)
            return ker;
    }
    return EnumT::undefined; // TODO: enum type must have undefined.
}

/**
 * Parses and returns a heterogeneous SSSP schedule.
 * Parameters:
 *   - fname <- name of the schedule file.
 * Returns:
 *   A heterogeneous SSSP schedule.
 */
SSSPHeteroSchedule load_sssp_schedule(std::string fname) {
    // Open file.
    std::ifstream ifs(fname, std::ifstream::in);

    // Initalize schedule.
    SSSPHeteroSchedule schedule;
    std::unordered_map<Device, int> devid_map; // Device -> Allocated ID map.
    Device cur_device;      // Current device.
    int    cur_device_id;   // Current device id.
    int    max_segment = 0; // Maximum segment ID.
    
    std::string line;
    while (std::getline(ifs, line)) {
        Device dev = get_device(line);
        // If start of new device.
        if (dev != Device::undefined) {
            cur_device    = dev;
            cur_device_id = devid_map[cur_device]++;

        // Otherwise, continue logging segments.
        } else {
            // Get range.
            auto space_idx = line.find_first_of(" ");
            int  start_seg = std::stoi(line.substr(0, space_idx));
            int  end_seg   = std::stoi(line.substr(space_idx + 1, 
                        std::string::npos));

            // Get kernel.
            std::getline(ifs, line); // Get next line.
            switch (get_device_type(cur_device)) {
                case DeviceType::CPU: {
                    // Get kernel name.
                    SSSPCPU ker = get_kernel_id<SSSPCPU>(line);

                    // Push to block list.
                    SSSPCPUScheduleBlock block(cur_device_id, 
                            start_seg, end_seg, get_kernel(ker), ker);
                    schedule.blocks.push_back(std::move(block));
                    break;
                }
                case DeviceType::GPU: {
                    // Get kernel name.
                    SSSPGPU ker = get_kernel_id<SSSPGPU>(line);

                    // Get block count and thread count.
                    int ker_str_len = to_string(ker).length();
                    std::string bctc = line.substr(ker_str_len + 1, 
                            std::string::npos);
                    space_idx = bctc.find_first_of(" ");
                    int block_count  = std::stoi(bctc.substr(0, space_idx));
                    int thread_count = std::stoi(bctc.substr(space_idx + 1, 
                            std::string::npos));

                    // Push to block list.
                    SSSPGPUScheduleBlock block(cur_device_id, 
                            start_seg, end_seg, get_kernel(ker), ker, 
                            block_count, thread_count);
                    schedule.blocks.push_back(std::move(block));
                    break;
                }
                default: {
                    std::cerr << "Cannot parse for undefined device type." 
                        << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }

            // Update data structures.
            max_segment = std::max(max_segment, end_seg);
        }
    }

    schedule.num_segments = max_segment + 1; // Segment ranges are inclusive.

    return schedule;
}

#endif // SRC__SCHEDULER_H
