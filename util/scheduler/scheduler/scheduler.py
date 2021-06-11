import copy
from typing import *

class Scheduler:
    def __init__(self, profiles: List[dict]):
        self.profiles = profiles
        print(profiles)

##

#def schedule(profile1, profile2, profile3, seg=0, exec1=0, exec2=0, exec3=0):
    ### Base case.
    ### len(profile1) == len(profile2) == len(profile3)
    #if seg == len(profile1):
        #return (max(exec1, exec2, exec3), (exec1, exec2, exec3))

    #min_profile = (float('inf'), (0, 0, 0))
    
    #(min_exec, execs) = schedule(profile1, profile2, profile3, seg + 1, 
                                 #exec1 + profile1[seg],
                                 #exec2,
                                 #exec3)
    #if min_exec < min_profile[0]:
        #min_profile = (min_exec, execs)

    #(min_exec, execs) = schedule(profile1, profile2, profile3, seg + 1, 
                                 #exec1,
                                 #exec2 + profile2[seg],
                                 #exec3)
    #if min_exec < min_profile[0]:
        #min_profile = (min_exec, execs)

    #(min_exec, execs) = schedule(profile1, profile2, profile3, seg + 1, 
                                 #exec1,
                                 #exec2,
                                 #exec3 + profile3[seg])
    #if min_exec < min_profile[0]:
        #min_profile = (min_exec, execs)

    #return min_profile

def schedule(profiles, execs, segments, seg=0):
    """Returns the best configuration where the maximum time between all devices
    is minimized."""
    # Base case.
    # len(profiles[0]) == len(profiles[1]) == ... == len(profiles[n - 1])
    if seg == len(profiles[0]):
        return (max(*execs), copy.copy(execs), copy.deepcopy(segments))

    ## Recursive case.
    min_profile = (float('inf'), [], [])

    for i, profile in enumerate(profiles):
        execs[i] += profile[seg] # What happens if we give device i this segment.
        segments[i].append(seg)
        min_res = schedule(profiles, execs, segments, seg + 1)
        execs[i] -= profile[seg] # Restore original execution times.
        segments[i].pop()
        
        if min_res[0] < min_profile[0]:
            min_profile = min_res

    return min_profile
            
profile1 = [10, 30, 15, 17]
profile2 = [30, 20, 10, 50]
profile3 = [25, 5, 30, 25]
profiles = [profile1, profile2, profile3]
init_execs = [0] * 3 # Initial execution times.
init_segments = [[] for _ in range(3)]
print(schedule(profiles, init_execs, init_segments))
