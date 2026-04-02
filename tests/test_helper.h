

#include <cmath>
#include <cstdlib>
#include <iostream>


inline void fail(const char *msg)
{
    std::cerr << msg << "\n";
    std::exit(1);
}


inline void expect_close(float a, float b, float tolerance, const char *msg)
{
    if (std::fabs(a - b) > tolerance) {
        std::cerr
            << msg
            << ": got=" << a
            << " expected=" << b
            << " tol=" << tolerance
            << "\n";
        std::exit(1);
    }
}

