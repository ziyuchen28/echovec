
#include <cstddef>
#include <iostream>
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


// optimization 1: 4 acc, loop unroll, make use of most ymm regisers (but below 16 to avoid ymm spilling) 
float dot_x86_avx2_4acc(const float *a, const float *b, std::size_t n) noexcept
{
    __m256 acc0 = _mm256_setzero_ps(); // ymm0
    __m256 acc1 = _mm256_setzero_ps(); // ymm1
    __m256 acc2 = _mm256_setzero_ps(); // ymm2
    __m256 acc3 = _mm256_setzero_ps(); // ymm3

    std::size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        // ymm4 - ymm7
        const __m256 va0 = _mm256_loadu_ps(a + i);
        const __m256 va1 = _mm256_loadu_ps(a + i + 8);
        const __m256 va2 = _mm256_loadu_ps(a + i + 16);
        const __m256 va3 = _mm256_loadu_ps(a + i + 24);

        // ymm8 - ymm11
        const __m256 vb0 = _mm256_loadu_ps(b + i);
        const __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        const __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        const __m256 vb3 = _mm256_loadu_ps(b + i + 24);

        // trigger FMA
        // vfmadd231ps ymm0, ymm4, ymm8
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        // vfmadd231ps ymm1, ymm5, ymm9
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
        // vfmadd231ps ymm2, ymm6, ymm10
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(va2, vb2));
        // vfmadd231ps ymm3, ymm7, ymm11
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(va3, vb3));
    }

    // vaddps ymm0, ymm0, ymm1
    // vaddps ymm2, ymm2, ymm3
    // vaddps ymm0, ymm0, ymm2
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


static float hsum256_ps(__m256 v) noexcept
{
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, sum);
    sum = _mm_add_ps(sum, sum);
    return _mm_cvtss_f32(sum);

}


// optimization 1: 4 acc, loop unroll, make use of most ymm regisers (but below 16 to avoid ymm spilling) 
// optimization 2: in register hotizontal sum, strictly kept in register, avoid cpu cache store load cycles
float dot_x86_avx2_4acc_inreghsum(const float *a, const float *b, std::size_t n) noexcept
{
    __m256 acc0 = _mm256_setzero_ps(); // ymm0
    __m256 acc1 = _mm256_setzero_ps(); // ymm1
    __m256 acc2 = _mm256_setzero_ps(); // ymm2
    __m256 acc3 = _mm256_setzero_ps(); // ymm3

    std::size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        // ymm4 - ymm7
        const __m256 va0 = _mm256_loadu_ps(a + i);
        const __m256 va1 = _mm256_loadu_ps(a + i + 8);
        const __m256 va2 = _mm256_loadu_ps(a + i + 16);
        const __m256 va3 = _mm256_loadu_ps(a + i + 24);

        // ymm8 - ymm11
        const __m256 vb0 = _mm256_loadu_ps(b + i);
        const __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        const __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        const __m256 vb3 = _mm256_loadu_ps(b + i + 24);

        // trigger FMA
        // vfmadd231ps ymm0, ymm4, ymm8
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        // vfmadd231ps ymm1, ymm5, ymm9
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
        // vfmadd231ps ymm2, ymm6, ymm10
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(va2, vb2));
        // vfmadd231ps ymm3, ymm7, ymm11
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(va3, vb3));
    }

    // vaddps ymm0, ymm0, ymm1
    // vaddps ymm2, ymm2, ymm3
    // vaddps ymm0, ymm0, ymm2
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));


    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }


    float sum = hsum256_ps(acc);

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}


float dot_x86_avx2(const float *a, const float *b, std::size_t n) noexcept
{
    // no optimization
    // reurn dot_x86_avx2_baseline(a, b, n); 

    // o1 - optimization level 1: loop unrolled to make full use of all 16 ymm registers
    // return dot_x86_avx2_4acc(a, b, n);

    // o2 - optimiation level 2: loop unrolled + in-register reduction
    return dot_x86_avx2_4acc_inreghsum(a, b, n);
}




} // namespace vecstore::vecmath



