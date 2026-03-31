#include "vecstore/vecmath/dot.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string get_arg(int argc, char **argv, const std::string &key, const std::string &def)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }

    return def;
}

vecstore::vecmath::DotImpl parse_impl(const std::string &s)
{
    using vecstore::vecmath::DotImpl;
    if (s == "scalar") {
        return DotImpl::Scalar;
    }

    if (s == "avx2") {
        return DotImpl::Avx2;
    }
    return DotImpl::Auto;
}

} // namespace


int main(int argc, char **argv)
{
    const std::size_t dim =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--dim", "1536")));

    const std::size_t iters =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--iters", "200000")));

    const std::size_t warmup =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--warmup", "1000")));

    const auto requested_impl =
        parse_impl(get_arg(argc, argv, "--impl", "auto"));

    const auto resolved_impl =
        vecstore::vecmath::resolve_dot_impl(requested_impl);

    const auto fn =
        vecstore::vecmath::resolve_dot_function(requested_impl);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a(dim);
    std::vector<float> b(dim);

    for (std::size_t i = 0; i < dim; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    volatile float sink = 0.0f;

    for (std::size_t i = 0; i < warmup; ++i) {
        sink += fn(a.data(), b.data(), dim);
    }

    const auto t0 = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < iters; ++i) {
        sink += fn(a.data(), b.data(), dim);
    }

    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per_op = sec * 1e9 / static_cast<double>(iters);
    const double ops_per_sec = static_cast<double>(iters) / sec;
    const double bytes_per_op = static_cast<double>(2 * dim * sizeof(float));
    const double gb_per_sec = (bytes_per_op * ops_per_sec) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "benchmark=dot\n";
    std::cout << "requested_impl=" << vecstore::vecmath::dot_impl_name(requested_impl) << "\n";
    std::cout << "resolved_impl=" << vecstore::vecmath::dot_impl_name(resolved_impl) << "\n";
    std::cout << "dim=" << dim << "\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "seconds=" << sec << "\n";
    std::cout << "ns_per_op=" << ns_per_op << "\n";
    std::cout << "ops_per_sec=" << ops_per_sec << "\n";
    std::cout << "effective_gb_per_sec=" << gb_per_sec << "\n";
    std::cout << "sink=" << sink << "\n";

    return 0;
}
