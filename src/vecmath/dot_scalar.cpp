#include "vecstore/vecmath/dot.h"

namespace vecstore::vecmath {

float dot_scalar(const float *a, const float* b, std::size_t n) noexcept {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float dot(const float *a, const float *b, std::size_t n) noexcept {
    return dot_scalar(a, b, n);
}

} // namespace vecstore::vecmath
