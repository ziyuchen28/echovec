#pragma once

#include <cstddef>

namespace vecstore::vecmath {

// pure scalar baseline
float dot_scalar(const float *a, const float *b, std::size_t n) noexcept;

float dot(const float *a, const float *b, std::size_t n) noexcept;

} // namespace vecstore::vecmath
