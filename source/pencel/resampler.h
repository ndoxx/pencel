#pragma once

#include <cstdint>
#include <string>

#define BLOCK_OFFSET_RGB24(ptr, width, x, y) (ptr + (3 * width) * y + 3 * x)

// * Following code is a partial port of Ramenhut's image resampler
// Original is under BSD-2-Clause licence
// https://github.com/ramenhut/single-header-image-resampler
namespace rs
{

enum KernelType : uint8_t
{
    KernelTypeUnknown,
    KernelTypeNearest,
    KernelTypeAverage,
    KernelTypeBilinear,
    KernelTypeBicubic,
    KernelTypeMitchell,
    KernelTypeCardinal,
    KernelTypeBSpline,
    KernelTypeLanczos,
    KernelTypeLanczos2,
    KernelTypeLanczos3,
    KernelTypeLanczos4,
    KernelTypeLanczos5,
    KernelTypeCatmull,
    KernelTypeGaussian,
};

bool ResampleImage24(uint8_t* src, uint32_t src_width, uint32_t src_height, uint8_t* dst, uint32_t dst_width,
                     uint32_t dst_height, KernelType type, std::string* errors = nullptr);

} // namespace rs
