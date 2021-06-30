/**
 * Schedule file loading and parsing.
 */

#ifndef SRC__SCHEDULER_H
#define SRC__SCHEDULER_H

#include <fstream>
#include <iostream> // TEMP
#include <string>

#include "devices.h"

/******************************************************************************
 ***** Data Structures ********************************************************
 ******************************************************************************/

/******************************************************************************
 ***** Helper Functions *******************************************************
 ******************************************************************************/

void load_schedule(std::string fname) {
    std::ifstream ifs(fname, std::ifstream::in);
    
    std::string line;
    while (std::getline(ifs, line)) {
        Device dev = get_device(line);
        // If start of new device.
        if (dev != Device::undefined) {
            std::cout << "Is device! " << dev << std::endl;
        // Otherwise, continue logging segments.
        } else {
            std::cout << "Logging segment " << line.substr(3) << std::endl;
        }
    }
}

#endif // SRC__SCHEDULER_H
