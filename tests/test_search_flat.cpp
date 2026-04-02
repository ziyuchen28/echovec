#include "vecstore/flat/search.h"
#include "vecstore/vecmath/normalize.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace vecstore::flat;
using namespace vecstore::vecmath;

void fail(const char *msg)
{
    std::cerr << msg << "\n";
    std::exit(1);
}


void expect_close(float a, float b, float tol, const char *msg)
{
    if (std::fabs(a - b) > tol) {
        std::cerr
            << msg
            << ": got=" << a
            << " expected=" << b
            << " tol=" << tol
            << "\n";
        std::exit(1);
    }
}


void test_small_known_case()
{
    const std::size_t dim = 4;
    const std::size_t db_rowcount = 4;
    const std::size_t k = 2;

    // could potentially be large and hence live in dynamic memory
    const std::vector<float> db {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 0.0f
    };

    const float query[4] {1.0f, 0.0f, 0.0f, 0.0f};

    std::vector<SearchResult> out;

    search_topk(
        db.data(),
        db_rowcount,
        dim,
        query,
        k,
        out,
        DotImpl::Scalar
    );

    if (out.size() != 2) {
        fail("test_small_known_case: expected 2 results");
    }

    if (out[0].index != 0) {
        fail("test_small_known_case: expected top-1 index == 0");
    }

    if (out[1].index != 3) {
        fail("test_small_known_case: expected top-2 index == 3");
    }

    expect_close(out[0].score, 1.0f, 1e-6f, "top-1 score mismatch");
    expect_close(out[1].score, 0.5f, 1e-6f, "top-2 score mismatch");
}


void test_scalar_vs_auto()
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::size_t count = 256;
    const std::size_t dim = 1024;
    const std::size_t k = 5;

    std::vector<float> db(count * dim);
    std::vector<float> query(dim);

    for (float &x : db) {
        x = dist(rng);
    }

    for (float &x : query) {
        x = dist(rng);
    }

    std::vector<SearchResult> scalar_out;
    std::vector<SearchResult> auto_out;

    search_topk(
        db.data(),
        count,
        dim,
        query.data(),
        k,
        scalar_out,
        DotImpl::Scalar
    );

    search_topk(
        db.data(),
        count,
        dim,
        query.data(),
        k,
        auto_out,
        DotImpl::Auto
    );

    if (scalar_out.size() != auto_out.size()) {
        fail("test_scalar_vs_auto: result size mismatch");
    }

    for (std::size_t i = 0; i < scalar_out.size(); ++i) {
        if (scalar_out[i].index != auto_out[i].index) {
            fail("test_scalar_vs_auto: top-k index mismatch");
        }

        expect_close(
            scalar_out[i].score,
            auto_out[i].score,
            1e-3f,
            "test_scalar_vs_auto: top-k score mismatch"
        );
    }
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


} // namespace


int main()
{
    test_small_known_case();
    test_scalar_vs_auto();

    std::cout << "OK: test_search\n";
    return 0;
}


