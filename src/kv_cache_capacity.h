#pragma once

// Internal KV capacity/view helpers for append-only PreciseSDPA caches.
// Not part of the public HunyuanOCR-ncnn API.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <mat.h>

namespace hunyuan_ocr {
namespace detail {

// Growth quantum for capacity-bearing KV backing storage (rows).
constexpr int kKvCacheCapacityChunkRows = 128;

// Safe capacity in rows for a 3D fp32 pack1 Mat.
// Uses cstep/w only when cstep is an exact multiple of w so partial-row padding
// is never counted as capacity. Returns 0 when layout is incompatible.
inline int kv_cache_capacity_rows(const ncnn::Mat& mat)
{
    if (mat.dims != 3 || mat.elemsize != 4u || mat.elempack != 1 || mat.w <= 0)
    {
        return 0;
    }
    if (mat.cstep <= 0 || (mat.cstep % static_cast<size_t>(mat.w)) != 0)
    {
        return 0;
    }
    const size_t rows = mat.cstep / static_cast<size_t>(mat.w);
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return 0;
    }
    return static_cast<int>(rows);
}

inline int round_up_kv_capacity_rows(int rows)
{
    if (rows <= 0)
    {
        return 0;
    }
    const int chunk = kKvCacheCapacityChunkRows;
    if (rows > std::numeric_limits<int>::max() - (chunk - 1))
    {
        return 0;
    }
    return ((rows + chunk - 1) / chunk) * chunk;
}

// Shallow view over past with logical h set to logical_rows.
// Keeps data/refcount/cstep so remaining capacity stays addressable.
// Requires capacity_rows(past) >= logical_rows and past layout compatible.
bool make_kv_cache_logical_view(const ncnn::Mat& past,
                                int logical_rows,
                                ncnn::Mat* output,
                                std::string* error);

// Allocate capacity-bearing 3D fp32 pack1 cache (h=capacity_rows in create),
// then set logical h to logical_rows. capacity_rows must be >= logical_rows.
bool allocate_kv_cache_with_capacity(int width,
                                     int logical_rows,
                                     int capacity_rows,
                                     int channels,
                                     ncnn::Mat* output,
                                     std::string* error,
                                     ncnn::Allocator* allocator = nullptr);

// Copy [0, rows) from source into dest starting at dest_row_offset for every
// channel. Both mats must be 3D fp32 pack1 with matching w/c.
bool copy_kv_cache_rows(const ncnn::Mat& source,
                        int source_row,
                        int rows,
                        ncnn::Mat* dest,
                        int dest_row,
                        std::string* error);

// Append current onto past into *key/*value for PreciseSDPA:
// - If past has capacity: no-copy past, only memcpy current after logical past.
// - Else: grow to round_up(past.h+current.h, 128), copy past+current once.
// Single-caller decode may share read-only past rows with the input past Mat;
// concurrent writers to the same backing storage are not supported.
bool append_or_grow_kv_cache(const ncnn::Mat& past_key,
                             const ncnn::Mat& past_value,
                             const ncnn::Mat& current_key,
                             const ncnn::Mat& current_value,
                             ncnn::Mat* key,
                             ncnn::Mat* value,
                             std::string* error,
                             ncnn::Allocator* allocator = nullptr);

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)

struct KvCacheAppendProfile {
    double allocation_ms = 0.0;
    double copy_ms = 0.0;
    std::uint64_t past_copy_bytes = 0;
    std::uint64_t current_copy_bytes = 0;
};

// Profile-only variant with real allocation/copy timing. The regular helper
// contains no timing branches when profiling is disabled.
bool append_or_grow_kv_cache_profiled(const ncnn::Mat& past_key,
                                      const ncnn::Mat& past_value,
                                      const ncnn::Mat& current_key,
                                      const ncnn::Mat& current_value,
                                      ncnn::Mat* key,
                                      ncnn::Mat* value,
                                      KvCacheAppendProfile* profile,
                                      std::string* error,
                                      ncnn::Allocator* allocator = nullptr);

#endif

} // namespace detail
} // namespace hunyuan_ocr
