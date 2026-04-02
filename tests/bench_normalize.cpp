
#include "vecstore/vecmath/normalize.h"
#include "vecstore/flat/search.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>


std::string get_arg(int argc, char **argv, const std::string &key, const std::string &def)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }

    return def;
}



void test_cosine_search_via_normalization()
{
    const std::size_t dim = 2;
    const std::size_t count = 3;

    // Without normalization:
    //   q dot v0 = 100
    //   q dot v1 = 2
    //   q dot v2 = 50
    // dot search prefers v0.
    //
    // With L2 normalization:
    //   cosine(q, v1) = 1
    //   cosine(q, v0) ~ 0.707
    //   cosine(q, v2) ~ 0.707
    // cosine search should prefer v1

    std::vector<float> db {
        100.0f, 0.0f,   // v0
        1.0f,   1.0f,   // v1
        0.0f,  50.0f    // v2
    };

    float query[2] {1.0f, 1.0f};

    std::vector<vecstore::flat::SearchResult> dot_out;
    std::vector<vecstore::flat::SearchResult> cosine_out;

    vecstore::flat::search_topk(
        db.data(),
        count,
        dim,
        query,
        1,
        dot_out,
        vecstore::vecmath::DotImpl::Scalar
    );

    if (dot_out.empty() || dot_out[0].index != 0) {
        fail("test_cosine_search_via_normalization: expected raw dot top-1 == v0");
    }

    vecstore::vecmath::normalize_rows_l2_inplace(db.data(), count, dim);
    vecstore::vecmath::normalize_l2_inplace(query, dim);

    vecstore::flat::search_topk(
        db.data(),
        count,
        dim,
        query,
        1,
        cosine_out,
        vecstore::vecmath::DotImpl::Scalar
    );

    if (cosine_out.empty() || cosine_out[0].index != 1) {
        fail("test_cosine_search_via_normalization: expected cosine top-1 == v1");
    }
}


int main(int argc, char **argv)
{

    const std::size_t db_rowcout =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--count", "20000")));

    const std::size_t dim =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--dim", "1536")));

    const std::size_t iters =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--iters", "100000")));

    const std::size_t warmup =
        static_cast<std::size_t>(std::stoull(get_arg(argc, argv, "--warmup", "100")));
}
