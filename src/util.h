/** 
 * Utilities.
 */

#ifndef SRC__UTIL_H
#define SRC__UTIL_H

#include <cstdlib>
#include <iostream>

#define CONDCHK( cond, msg ) {\
    if (cond) {\
        std::cerr << "(" << __FILE__ << ", " << __LINE__ << ") " << msg \
            << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

#include "../deps/gapbs/src/timer.h"

#endif // SRC__UTIL_H
