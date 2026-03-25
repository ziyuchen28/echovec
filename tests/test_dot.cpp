#include "vecstore/vecmath/dot.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {


double dot_ref(const float *a, const float *b, std::size_t n) 
{
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}


void test_small_exact() 
{
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 5.0f, 6.0f};

    const float got = vecstore::vecmath::dot(a.data(), b.data(), a.size());
    const float expected = 32.0f;

    if (std::fabs(got - expected) > 1e-6f) {
        std::cerr << "small exact failed: got=" << got
            << " expected=" << expected << "\n";
        std::exit(1);
    }
}


void test_random_vs_reference() 
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t n = 1536; // 1536 - easier unrolling
    std::vector<float> a(n), b(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    const float got = vecstore::vecmath::dot(a.data(), b.data(), n);
    const double ref = dot_ref(a.data(), b.data(), n);

    const double abs_err = std::fabs(static_cast<double>(got) - ref);
    if (abs_err > 1e-3) {
        std::cerr << "random reference failed: got=" << got
            << " ref=" << ref
            << " abs_err=" << abs_err << "\n";
        std::exit(1);
    }
}

} // namespace


int main() 
{
    test_small_exact();
    test_random_vs_reference();
    std::cout << "OK: test_dot\n";
    return 0;
}
