#include "vecstore/vecmath/normalize.h"

#include <cmath>


namespace vecstore::vecmath {

#define f32 float
#define f64 double


static f32 l2_norm(const f32 *v, std::size_t n) noexcept
{
    f64 sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        // avoid overflowing to infinity
        const f64 x = static_cast<f64>(v[i]);
        // to do - SIMD optimization, for now trading safty (64) for throughput
        // as this is a one time computation
        sum += x * x;
    }
    return static_cast<f32>(std::sqrt(sum));
}


bool normalize_l2_inplace(f32 *v, std::size_t n) noexcept
{
    const f32 norm = l2_norm(v, n);
    if (norm <= 0.0f) {
        return false;
    }
    const f32 inverse = 1.0f / norm;
    for (std::size_t i = 0; i < n; ++i) {
        // prefer multiplication over division for fewer cpu cycles
        v[i] *= inverse;
    }
    return true;
}


void normalize_rows_l2_inplace(f32 *db, std::size_t count, std::size_t dim) noexcept
{
    for (std::size_t row = 0; row < count; ++row) {
        normalize_l2_inplace(db + row * dim, dim);
    }
}

} // namespace vecstore::vecmath
