#include "vecstore/vecmath/dot.h"


namespace vecstore::vecmath {

#if defined(VECSTORE_BUILD_X86_AVX2)
// float dot_x86_avx2_baseline(const float *a, const float *b, std::size_t n) noexcept;
float dot_x86_avx2_unrolled(const float *a, const float *b, std::size_t n) noexcept;
#endif


#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define VECSTORE_X86_RUNTIME_DETECT 1
#else
#define VECSTORE_X86_RUNTIME_DETECT 0
#endif


// pick the best available
DotImpl detect_best_dot_impl() noexcept
{
#if VECSTORE_X86_RUNTIME_DETECT && defined(VECSTORE_BUILD_X86_AVX2) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        return DotImpl::Avx2;
    }
#endif
    return DotImpl::Scalar;
}


DotImpl resolve_dot_impl(DotImpl impl) noexcept
{
    if (impl == DotImpl::Auto) {
        return detect_best_dot_impl();
    }

    if (impl == DotImpl::Avx2) {
#if defined(VECSTORE_BUILD_X86_AVX2)
        if (detect_best_dot_impl() == DotImpl::Avx2) {
            return DotImpl::Avx2;
        }
#endif
        return DotImpl::Scalar;
    }
    return DotImpl::Scalar;
}


DotFn resolve_dot_function(DotImpl impl) noexcept
{
    const DotImpl resolved = resolve_dot_impl(impl);
    switch (resolved) {
        case DotImpl::Avx2: {
#if defined(VECSTORE_BUILD_X86_AVX2)
            return &dot_x86_avx2_unrolled;
            // return &dot_x86_avx2_baseline;
#else
            return &dot_scalar;
#endif
        }
        case DotImpl::Scalar:
        case DotImpl::Auto:
        default: {
            return &dot_scalar;
        }
    }
}


const char *dot_impl_name(DotImpl impl) noexcept
{
    switch (impl) {
        case DotImpl::Scalar:
            return "scalar";
        case DotImpl::Avx2:
            return "avx2";
        case DotImpl::Auto:
        default:
            return "auto";
    }
}


float dot(const float *a, const float *b, std::size_t n, DotImpl impl) noexcept
{
    const DotFn fn = resolve_dot_function(impl);
    return fn(a, b, n);
}

} // namespace vecstore::vecmath


