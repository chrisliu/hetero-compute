/** 
 * Utilities.
 */

#ifndef SRC__UTIL_H
#define SRC__UTIL_H

#include <cstdlib>
#include <iostream>
#include <type_traits>

/** Timer */
#include "../deps/gapbs/src/timer.h"

/** Condition check. */
#define CONDCHK( cond, msg ) {\
    if (cond) {\
        std::cerr << "(" << __FILE__ << ", " << __LINE__ << ") " << msg \
            << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

#define UNUSED __attribute__((unused))

/** Type traits. */

// Is type a member of some defined set of types.
// Usage: is_contained<TypeToCheck, SetType1, SetType2, ... , SetTypeN>
// https://stackoverflow.com/questions/16252902/sfinae-set-of-types-contains-the-type
template <typename T, typename ...> struct is_contained : std::false_type {};
template <typename T, typename Head, typename ...Tail>
struct is_contained<T, Head, Tail...> : std::integral_constant<bool,
    std::is_same<T, Head>::value || is_contained<T, Tail...>::value> {};

// Is type an instance of a templated instance of a class.
// Usage: is_templated_instance<CheckT, TemplatedClassT>
// https://stackoverflow.com/questions/44012938/how-to-tell-if-template-type-is-an-instance-of-a-template-class
template <typename, template <typename, typename...> typename>
struct is_templated_instance : public std::false_type {};
template <typename...Ts, template <typename, typename...> typename U>
struct is_templated_instance<U<Ts...>, U> : public std::true_type {};

#endif // SRC__UTIL_H
