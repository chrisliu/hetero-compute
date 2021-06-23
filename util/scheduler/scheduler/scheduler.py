# Variety of scheduler implementations and data structures.

from __future__ import annotations
import copy
from typing import *

################################################################################
##### Data Structures ##########################################################
################################################################################

class KernelSegment:
    """POD that contains a kernel and a segment."""

    def __init__(self, kernel_name: str, segment: int, exec_time: float):
        self.kernel_name = kernel_name
        self.segment     = segment
        self.exec_time   = exec_time

    def __repr__(self) -> str:
        return f'KernelSegment(kernel={self.kernel_name}, '\
            f'segment={self.segment})'

    def __eq__(self, other) -> bool:
        return self.kernel_name == other.kernel_name and \
            self.segment == other.segment

class DeviceSchedule:
    """Schedule for a single device."""
    
    def __init__(self, device_name: str):
        self.device_name: str                 = device_name
        self.exec_time  : float               = float(0)
        self.schedule   : List[KernelSegment] = list()

    def __repr__(self) -> str:
        return f'DeviceSchedule(device={self.device_name})'

################################################################################
##### Schedulers ###############################################################
################################################################################

class Scheduler:
    def schedule(self, config: Dict[str, int], metric: Metric) \
            -> List[DeviceSchedule]:
        raise NotImplemented("schedule not implemented.")

class ExhaustiveScheduler(Scheduler):
    """Exhausitively search all permutations, returning the best configuration
    based on some metric."""

    def __init__(self, profiles: List[DeviceProfile]):
        self.profiles = profiles

    def schedule(self, config: Dict[str, int], metric: Metric) \
            -> List[DeviceSchedule]:
        # Initialize devices.
        schedules: Dict[str, DeviceSchedule] = {
            device_name: [DeviceSchedule(device_name) for _ in range(count)]
            for (device_name, count) in config.items()
        }

        # Get optimal schedule.
        (_, schedules) = self.__schedule_impl(schedules, metric)
        
        # Return flattened schedule.
        flat_schedules = [sched for (_, sched_list) in schedules.items()
                         for sched in sched_list]
        return flat_schedules

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

#class GreedyScheduler(Scheduler):
    #"""Greedily search for an optimal configuration."""

    #class BFSSolver:
        #"""BFS-based greedy solver.

        #Graph structure:
          #Each segment represents a layer of nodes. Each node represents a 
          #particular device right before this segment. The edges determines
          #which device+kernel solves the current segment.
        #"""
        #def __init__(self, profiles: List[DeviceProfile], 
                     #config: Dict[str, int]):
            ## Construct Dict[device name, List[kernel profiles]]
            #self.profiles = {
                #prof.device_name: pro.kernel_profiles
                #for prof in profiles
            #}

            #self.config = config

            ## Initialize frontier.
            ## Is a Dict[device name, List[Tuple(subdevice id, best metric)]]
            #self.frontier = None

        #def solve(self, metric: Metric, best_n = 1) \
                #-> Dict[str, DeviceSchedule]:
            #"""Uses BFS to greedily searches for a best solution given a 
            #particular metric.

            #Parameters:
              #- metric <- metric to use.
              #- best_n <- initialize best device with <best_n> segments.
            #Returns:
              #best schedule.
            #"""

            #self.frontier = self.__init_frontier(metric)

            #best_device_results = {
                #devname: sum(min(kerprof[i] for kerprof in kerprofiles)
                             #for i in range(best_n))
                #for devname, kerprofiles in self.profiles.items()
            #}
            #best_device = min(best_device_results.items(),
                              #key = lambda res: res[1])



        #def __init_frontier(self, metric: Metric) \
                #-> Dict[str, List[Tuple[int, Metric]]]:
            #frontier = dict()
            #for devname, count in self.config.items():
                #frontier[devname] = [(i, metric.default_metric())
                                     #for i in range(count)]
            #return frontier

        #def __best_device_for_init_n(self, best_n) -> DeviceSchedule:
            #pass

    #def __init__(self, profiles: List[DeviceProfile]):
        #self.profiles = profiles

    #def schedule(self, config: Dict[str, int], metric: Metric) \
            #-> List[DeviceSchedule]:
        ## Initialize devices.
        #schedules: Dict[str, DeviceSchedule] = {
            #device_name: [DeviceSchedule(device_name) for _ in range(count)]
            #for (device_name, count) in config.items()
        #}

        ## Get optimal schedule.
        #schedules = self.__schedule_impl(schedules, metric)
        
        ## Return flattened schedule.
        #flat_schedules = [sched for (_, sched_list) in schedules.items()
                         #for sched in sched_list]
        #return flat_schedules
    
    #def __schedule_impl(self, schedules: Dict[str, DeviceSchedule]) \
            #-> Dict[str, DeviceSchedule]:
        #pass
        
class MostGainScheduler:
    def __init__(self, profiles: List[DeviceProfile]):
        self.devices = [devprof.device_name for devprof in profiles]
        # Best kernel arrangement for each device.
        self.device_best = dict()
        self.device_exec = dict()

        # For each device.
        for devname, devprof in zip(self.devices, profiles):
            bestkers = list()
            exectime = 0

            # For each segment, get best result.
            for seg in range(len(devprof)):
                best = (float('inf'), '')
                for k in range(len(devprof.kernel_profiles)):
                    kerprof = devprof.kernel_profiles[k]
                    best = min(best, (kerprof[seg], kerprof.kernel_name))
                bestkers.append(KernelSegment(best[1], seg, best[0]))
                exectime += best[0]

            self.device_best[devname] = bestkers
            self.device_exec[devname] = exectime

        # Gains table. Gains[from device][to device].
        self.gains = dict()
        for devfrom in self.devices:
            self.gains[devfrom] = dict()
            for devto in self.devices:
                self.gains[devfrom][devto] = [
                    kerto.exec_time / kerfrom.exec_time
                    for (kerto, kerfrom) in zip(self.device_best[devto], 
                                                self.device_best[devfrom])
                ]

    def best_single_device_time(self) -> float:
        best_proc = min(self.device_exec.items(), key=lambda t: t[1])[0]
        return self.device_exec[best_proc]

    def schedule(self, config: Dict[str, int]) -> List[DeviceSchedule]:
        # Initialize devices.
        schedules: Dict[str, DeviceSchedule] = {
            device_name: [DeviceSchedule(device_name) for _ in range(count)]
            for (device_name, count) in config.items()
        }

        # Phase 1:
        #   Initialize schedule for device(s) that belong to the most effective
        #   processor.

        # Determine best processor overall.
        best_proc = min(self.device_exec.items(), key=lambda t: t[1])[0]
        
        # Roughly allot contiguous chunks to each of the best device.
        avg_exec = self.device_exec[best_proc] / config[best_proc]

        start_id = 0
        for devsched in schedules[best_proc]:
            for i, kerseg in enumerate(self.device_best[best_proc][start_id:]):
                devsched.exec_time += kerseg.exec_time
                devsched.schedule.append(kerseg)

                # Go to next device if time limit met.
                if devsched.exec_time >= avg_exec:
                    break

            start_id += i + 1

        # Rudimentary load balancing for processor.
        self.__balance_device(schedules, best_proc)

        ## Phase 2:
        ##   Greedily redistribute workloads to other device(s).
        while True:
            #pprint_schedule(self.__flatten_schedule(schedules))
            (okay, next_schedules) = self.__greedy_balance(schedules)
            if not okay: break
            schedules = next_schedules

        # Return flattened schedule.
        return self.__flatten_schedule(schedules)

    def __flatten_schedule(self, schedules: Dict[str, List[DeviceSchedule]]) \
            -> List[DeviceSchedule]:
        flat_schedules = [sched for (_, sched_list) in schedules.items()
                 for sched in sched_list]
        return flat_schedules

    def __get_worst_time(self, scheds: Dict[str, DeviceSchedule]) -> float:
        exec_times = [sched.exec_time
                      for (_, sched_list) in scheds.items()
                      for sched in sched_list]
        return max(exec_times)

    def __greedy_balance(self, scheds: Dict[str, List[DeviceSchedule]]) \
            -> Tuple[bool, Dict[str, List[DeviceSchedule]]]:
        # Find processor that takes the most time. Find processor that takes the
        # least time (that aren't the same processor).
        proc_times = list()
        for procname, devsched in scheds.items():
            total_exec = sum(d.exec_time for d in devsched)
            proc_times.append((total_exec / len(devsched), procname))
        proc_times = sorted(proc_times)

        slow_proc = proc_times[-1][1]

        # Create list of segments that belong to a processor.
        proc_segments = {
            procname: [(kerseg, i)
                       for i, dsched in enumerate(devsched)
                       for kerseg in dsched.schedule] 
            for procname, devsched in scheds.items()
        }
        proc_segments = {
            procname: sorted(segs, key=lambda ks: ks[0].segment)
            for procname, segs in proc_segments.items()
        }

        # Try to allocate to most available processor.
        old_time = self.__get_worst_time(scheds)

        for _, to_proc in proc_times[:-1]:
            # Give segment this @to_proc is best at relative to @slow_proc.
            worst_segment = (float('inf'), None)
            for kerseg in proc_segments[slow_proc]:
                cand_seg = (self.gains[slow_proc][to_proc][kerseg[0].segment], 
                            kerseg)
                worst_segment = min(worst_segment, cand_seg) # Least speedup.

            # Update schedule.
            nsched = copy.deepcopy(scheds)

            # Remove segment for @slow_proc.
            seg, devid = worst_segment[1]
            nsched[slow_proc][devid].schedule.remove(seg) 
            nsched[slow_proc][devid].exec_time -= seg.exec_time

            # Insert segment to @to_proc.
            insert_devid = 0
            insert_pos = 0
            for i, (segment, devid) in enumerate(proc_segments[to_proc]):
                if segment.segment > seg.segment: break
                if devid != insert_devid:
                    insert_devid = devid
                    insert_pos = 0
                else:
                    insert_pos += 1

            to_seg = self.device_best[to_proc][seg.segment]
            nsched[to_proc][insert_devid].schedule.insert(insert_pos, to_seg) 
            nsched[to_proc][insert_devid].exec_time += to_seg.exec_time

            # Load balance amongst devices.
            self.__balance_device(nsched, slow_proc)
            self.__balance_device(nsched, to_proc)

            # If worst time has been improved, return new schedule.
            if self.__get_worst_time(nsched) < old_time:
                return (True, nsched)

        return (False, None)

    def __balance_device(self, scheds: Dict[str, List[DeviceSchedule]], device):
        # Get average time.
        exec_times = [sched.exec_time for sched in scheds[device]]
        avg_time = sum(exec_times) / len(exec_times)

        def balance(from_dev, to_dev):
            # Old variance to beat.
            old_diff = (from_dev.exec_time - avg_time) ** 2 \
                + (to_dev.exec_time - avg_time) ** 2
            
            while True:
                from_seg = from_dev.schedule[-1]
                diff_time = from_seg.exec_time
                # Simulate moving lasts segment from @from_dev to @to_dev
                new_diff = (from_dev.exec_time - diff_time - avg_time) ** 2 \
                    + (to_dev.exec_time + diff_time - avg_time)

                # If detrimental effect, don't do it!
                if new_diff > old_diff: break

                del from_dev.schedule[-1]
                from_dev.exec_time -= diff_time
                to_dev.schedule.insert(0, from_seg)
                to_dev.exec_time += diff_time

                old_diff = new_diff

        # Go from left to right.
        for from_dev, to_dev in zip(scheds[device][:-1], scheds[device][1:]):
            if len(from_dev.schedule) == 0 or len(to_dev.schedule) == 0:
                continue
            balance(from_dev, to_dev)

        # Go from right to left.
        for from_dev, to_dev in \
                zip(scheds[device][1::-1], scheds[device][:-1:-1]):
            if len(from_dev.schedule) == 0 or len(to_dev.schedule) == 0:
                continue
            balance(from_dev, to_dev)

################################################################################
##### Helper functions #########################################################
################################################################################

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
