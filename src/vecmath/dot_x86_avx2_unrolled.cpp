
#include <cstddef>
#include <immintrin.h>

namespace vecstore::vecmath {


float dot_x86_avx2_unrolled(const float *a, const float *b, std::size_t n) noexcept
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);

        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);

        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(va2, vb2));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(va3, vb3));
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));


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
