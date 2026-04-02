#include "vecstore/flat/search.h"

#include <algorithm>
#include <utility>

namespace vecstore::flat {

#define f32 float
#define f64 double

namespace {

void insert_sorted_topk(std::vector<SearchResult> &out,
                        std::size_t k,
                        SearchResult candidate)
{
    if (out.size() == 0) {
        out.push_back(candidate);
        return;
    }
    if (out.size() < k) {
        out.push_back(candidate);

        std::size_t last = out.size() - 1;
        std::size_t pos = last - 1;
        while (pos > 0 && out[pos].score > out[last].score) {
            --pos;
        }
        std::swap(out[pos + 1], out[last]);
        return;
    }

    if (candidate.score <= out.back().score) {
        return;
    }

    out.back() = candidate;

    std::size_t last = out.size() - 1;
    std::size_t pos = last - 1;
    while (pos > 0 && out[pos].score > out[last].score) {
        --pos;
    }
    std::swap(out[pos + 1], out[last]);
}

} // namespace



using namespace vecstore::vecmath;

void search_topk(const f32 *db_inmemory,
                 std::size_t db_rowcount,
                 std::size_t dim,
                 const f32 *query,
                 std::size_t k,
                 std::vector<SearchResult> &out,
                 DotImpl impl)
{

    if (k == 0) return;
    out.clear();

    if (db_inmemory == nullptr || query == nullptr) {
        return;
    }

    if (db_rowcount == 0 || dim == 0 || k == 0) {
        return;
    }

    const std::size_t want = std::min(k, db_rowcount);
    if (out.capacity() < want) {
        out.reserve(want);
    }

    const DotFn dot_fn = resolve_dot_function(impl);

    for (std::size_t idx = 0; idx < db_rowcount; ++idx) {
        const f32 *vec = db_inmemory + idx * dim;
        const f32 score = dot_fn(vec, query, dim);
        insert_sorted_topk(out, want, SearchResult {idx, score});
    }
}

} // namespace vecstore::flat


