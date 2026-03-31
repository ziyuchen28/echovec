#include "vecstore/vecmath/dot.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <iostream>

namespace {


double dot_ref(const float *a, const float *b, std::size_t n) 
{
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}


void expect_close(double got, double expected, double tol, const char *label)
{
    const double err = std::fabs(got - expected);
    if (err > tol) {
        std::cerr
            << label
            << " failed: got=" << got
            << " expected=" << expected
            << " abs_err=" << err
            << " tolerance=" << tol
            << "\n";
        std::exit(1);
    }
}


void test_small_exact() 
{
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 5.0f, 6.0f};

    const float got_scalar = vecstore::vecmath::dot_scalar(a.data(), b.data(), a.size());
    const float got_auto = vecstore::vecmath::dot(a.data(), b.data(), a.size());
    const float expected = 32.0f;

    expect_close(got_scalar, expected, 1e-6, "small_exact scalar");
    expect_close(got_auto, expected, 1e-6, "small_exact auto");

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

    const float got_scalar = vecstore::vecmath::dot_scalar(a.data(), b.data(), a.size());
    const float got_auto = vecstore::vecmath::dot(a.data(), b.data(), a.size());
    
    const double ref = dot_ref(a.data(), b.data(), n);

    expect_close(got_scalar, ref, 1e-3, "random_ref scalar");
    expect_close(got_auto, ref, 1e-3, "random_ref auto");
}


void test_avx2()
{

    using namespace vecstore::vecmath;

    if (resolve_dot_impl(DotImpl::Avx2) != DotImpl::Avx2) {
        std:std::cerr << "[x] avx2 not available on this machine" << std::endl;
        return;
    }

    std::cout << "[*] avx2 available" << std::endl;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t n = 1024;

    std::vector<float> a(n);
    std::vector<float> b(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    const float scalar = dot_scalar(a.data(), b.data(), n);
    const float avx2 = dot(a.data(), b.data(), n, DotImpl::Avx2);

    expect_close(avx2, scalar, 1e-3, "forced_avx2");
}


} // namespace


int main() 
{
    test_small_exact();
    test_random_vs_reference();
    test_avx2();
    std::cout << "OK: test_dot\n";
    return 0;
}
