#pragma once

#include <cstddef>

namespace vecstore::vecmath
{

bool normalize_l2_inplace(float *v, std::size_t n) noexcept;

void normalize_rows_l2_inplace(float *base, std::size_t count, std::size_t dim) noexcept;

} // namespace vecstore::vecmath
