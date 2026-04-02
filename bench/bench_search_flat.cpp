#include "vecstore/flat/search.h"

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
    const std::size_t db_rowcout =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--count", "20000")));

    const std::size_t dim =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--dim", "1536")));

    const std::size_t k =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--k", "10")));

    const std::size_t iters =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--iters", "100")));

    const std::size_t warmup =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--warmup", "3")));

    const auto requested_impl =
        parse_impl(get_arg(argc, argv, "--impl", "auto"));

    const auto resolved_impl =
        vecstore::vecmath::resolve_dot_impl(requested_impl);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> db(db_rowcout * dim);
    std::vector<float> query(dim);

    for (float &x : db) {
        x = dist(rng);
    }

    for (float &x : query) {
        x = dist(rng);
    }

    std::vector<vecstore::flat::SearchResult> out;

    // sink: make sure search_topk actually run and not removed from compiler dead code elim
    // volaile: makle sure the loop runs iters time to avoid loop hoisting
    volatile float sink = 0.0f;

    for (std::size_t i = 0; i < warmup; ++i) {
        vecstore::flat::search_topk(
            db.data(),
            db_rowcout,
            dim,
            query.data(),
            k,
            out,
            requested_impl
        );

        if (!out.empty()) {
            sink += out[0].score;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < iters; ++i) {
        vecstore::flat::search_topk(
            db.data(),
            db_rowcout,
            dim,
            query.data(),
            k,
            out,
            requested_impl
        );

        if (!out.empty()) {
            sink += out[0].score;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per_query = sec * 1e9 / static_cast<double>(iters);
    const double qps = static_cast<double>(iters) / sec;
    const double vectors_per_sec = static_cast<double>(db_rowcout) * qps;

    // "effective" because data is hot/reused; this is useful for comparison,
    // not literal DRAM bandwidth.
    const double bytes_per_query = static_cast<double>(db_rowcout * dim * sizeof(float));
    const double effective_gb_per_sec = (bytes_per_query * qps) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "benchmark=flat_exact_search\n";
    std::cout << "requested_impl=" << vecstore::vecmath::dot_impl_name(requested_impl) << "\n";
    std::cout << "resolved_impl=" << vecstore::vecmath::dot_impl_name(resolved_impl) << "\n";
    std::cout << "db_rowcout=" << db_rowcout << "\n";
    std::cout << "dim=" << dim << "\n";
    std::cout << "k=" << k << "\n";
    std::cout << "iters=" << iters << "\n";
    std::cout << "seconds=" << sec << "\n";
    std::cout << "ns_per_query=" << ns_per_query << "\n";
    std::cout << "queries_per_sec=" << qps << "\n";
    std::cout << "vectors_scanned_per_sec=" << vectors_per_sec << "\n";
    std::cout << "effective_gb_per_sec=" << effective_gb_per_sec << "\n";
    std::cout << "top1_index=" << (out.empty() ? 0 : out[0].index) << "\n";
    std::cout << "top1_score=" << (out.empty() ? 0.0f : out[0].score) << "\n";
    std::cout << "sink=" << sink << "\n";

    return 0;
}


