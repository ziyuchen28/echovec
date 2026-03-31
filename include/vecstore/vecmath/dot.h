#pragma once

#include <cstddef>

namespace vecstore::vecmath {


enum class DotImpl
{
    Scalar,
    Avx2,
    Auto
};

using DotFn = float (*)(const float *a, const float *b, std::size_t n) noexcept;

// scalar baseline.
float dot_scalar(const float *a, const float *b, std::size_t n) noexcept;

// runtime detection of the best implementation available based on ISA.
DotImpl detect_best_dot_impl() noexcept;

// Resolve a requested implementation to what is actually usable.
// Example:
//   Auto  -> Avx2 or Scalar
//   Avx2  -> Avx2 if supported, otherwise Scalar
DotImpl resolve_dot_impl(DotImpl impl) noexcept;

// resolve a function pointer once, so benchmark loops don't measure dispatch overhead
DotFn resolve_dot_function(DotImpl impl = DotImpl::Auto) noexcept;

const char *dot_impl_name(DotImpl impl) noexcept;

// Convenience wrapper.
float dot(const float *a, const float *b, std::size_t n, DotImpl impl = DotImpl::Auto) noexcept;

} // namespace vecstore::vecmath
