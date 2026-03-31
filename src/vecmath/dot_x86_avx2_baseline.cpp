#include "vecstore/vecmath/dot.h"

#include <cstddef>
#include <immintrin.h>

namespace vecstore::vecmath {


float dot_x86_avx2_baseline(const float *a, const float *b, std::size_t n) noexcept
{
    __m256 acc = _mm256_setzero_ps();

    std::size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);

        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);

    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
        sum += tmp[lane];
    }

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

} // namespace vecstore::vecmath
