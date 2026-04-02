#pragma once

#include <cstddef>
#include <vector>

#include "vecstore/vecmath/dot.h"

namespace vecstore::flat {

struct SearchResult
{
    std::size_t index = 0;
    float score = 0.0f;
};


// db_inmemory: float 32 matrix [v0 | v1 | v2 | ...]
// out: top k descending score
void search_topk(const float *db_inmemory,
                 std::size_t db_rowcount,
                 std::size_t dim,
                 const float *query,
                 std::size_t k,
                 std::vector<SearchResult> &out,
                 vecstore::vecmath::DotImpl impl = vecstore::vecmath::DotImpl::Auto
);

} // namespace vecstore::flat
