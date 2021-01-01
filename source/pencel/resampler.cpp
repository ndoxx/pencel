#include "resampler.h"
#include <cstring>
#include <kibble/logger/logger.h>
#include <memory>

namespace rs
{

inline int32_t clip_range(int32_t input, int32_t low, int32_t high)
{
    return (input < low) ? low : (input > high) ? high : input;
}
enum KernelDirection : uint8_t
{
    KernelDirectionUnknown,
    KernelDirectionHorizontal,
    KernelDirectionVertical,
};

bool SampleKernelBilinearH(uint8_t* src, uint32_t src_width, uint32_t src_height, float f_x, float f_y, uint8_t* output)
{
    if(!src || !src_width || !src_height || f_x < 0 || f_y < 0 || !output)
    {
        return false;
    }

    /* We do not bias our float coordinate by 0.5 because we wish
       to sample using the nearest 2 pixels to our coordinate. */
    int32_t sample_x = int32_t(f_x);
    int32_t sample_y = int32_t(f_y);
    uint8_t* pixels[2] = {nullptr};
    float f_delta = float(f_x) - float(sample_x);

    /* compute our two pixels that will be interpolated together. */
    for(int32_t i = 0; i < 2; i++)
    {
        int32_t src_x = clip_range(sample_x + i, 0, int32_t(src_width) - 1);
        int32_t src_y = clip_range(sample_y, 0, int32_t(src_height) - 1);

        pixels[i] = BLOCK_OFFSET_RGB24(src, src_width, src_x, uint32_t(src_y));
    }

    /* perform the interpolation of our lerp_pixels. */
    output[0] = uint8_t(pixels[0][0] * (1.0f - f_delta) + pixels[1][0] * f_delta);
    output[1] = uint8_t(pixels[0][1] * (1.0f - f_delta) + pixels[1][1] * f_delta);
    output[2] = uint8_t(pixels[0][2] * (1.0f - f_delta) + pixels[1][2] * f_delta);

    return true;
}

bool SampleKernelBilinearV(uint8_t* src, uint32_t src_width, uint32_t src_height, float f_x, float f_y, uint8_t* output)
{
    if(!src || !src_width || !src_height || f_x < 0 || f_y < 0 || !output)
    {
        return false;
    }

    /* We do not bias our float coordinate by 0.5 because we wish
       to sample using the nearest 2 pixels to our coordinate. */
    int32_t sample_x = int32_t(f_x);
    int32_t sample_y = int32_t(f_y);
    uint8_t* pixels[2] = {nullptr};
    float f_delta = float(f_y) - float(sample_y);

    /* compute our two pixels that will be interpolated together. */
    for(int32_t i = 0; i < 2; i++)
    {
        int32_t src_x = clip_range(sample_x, 0, int32_t(src_width) - 1);
        int32_t src_y = clip_range(sample_y + i, 0, int32_t(src_height) - 1);

        pixels[i] = BLOCK_OFFSET_RGB24(src, src_width, src_x, uint32_t(src_y));
    }

    /* perform the interpolation of our lerp_pixels. */
    output[0] = uint8_t(pixels[0][0] * (1.0f - f_delta) + pixels[1][0] * f_delta);
    output[1] = uint8_t(pixels[0][1] * (1.0f - f_delta) + pixels[1][1] * f_delta);
    output[2] = uint8_t(pixels[0][2] * (1.0f - f_delta) + pixels[1][2] * f_delta);

    return true;
}

bool SampleKernelBilinear(uint8_t* src, uint32_t src_width, uint32_t src_height, KernelDirection direction, float f_x,
                          float f_y, uint8_t* output)
{
    switch(direction)
    {
    case KernelDirectionHorizontal:
        return SampleKernelBilinearH(src, src_width, src_height, f_x, f_y, output);
    case KernelDirectionVertical:
        return SampleKernelBilinearV(src, src_width, src_height, f_x, f_y, output);
    default:
        return false;
    }

    return false;
}

bool SampleKernel(uint8_t* src, uint32_t src_width, uint32_t src_height, KernelDirection direction, float f_x,
                  float f_y, KernelType type, float /*h_ratio*/, float /*v_ratio*/, uint8_t* output)
{
    if(type == KernelTypeBilinear)
        return SampleKernelBilinear(src, src_width, src_height, direction, f_x, f_y, output);
    KLOGW("pencel") << "Kernel type not implemented." << std::endl;
    return false;
}

bool ResampleImage24(uint8_t* src, uint32_t src_width, uint32_t src_height, uint8_t* dst, uint32_t dst_width,
                     uint32_t dst_height, KernelType type, ::std::string* errors)
{
    if(!src || !dst || !src_width || !src_height || !dst_width || !dst_height || type == KernelTypeUnknown)
    {
        if(errors)
        {
            *errors = "Invalid parameter passed to ResampleImage24.";
        }
        return false;
    }

    // uint32_t src_row_pitch = 3 * src_width;
    uint32_t dst_row_pitch = 3 * dst_width;
    uint32_t buffer_size = dst_row_pitch * src_height;
    uint32_t dst_image_size = dst_row_pitch * dst_height;

    if(src_width == dst_width && src_height == dst_height)
    {
        /* no resampling needed, simply copy the image over. */
        memcpy(dst, src, dst_image_size);
        return true;
    }

    ::std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);

    float h_ratio = (1 == dst_width ? 1.0f : (float(src_width) - 1) / float(dst_width - 1));
    float v_ratio = (1 == dst_height ? 1.0f : (float(src_height) - 1) / float(dst_height - 1));

    for(uint32_t j = 0; j < src_height; j++)
        for(uint32_t i = 0; i < dst_width; i++)
        {
            uint8_t* output = BLOCK_OFFSET_RGB24(buffer.get(), dst_width, i, j);

            float f_x = float(i) * h_ratio;
            float f_y = float(j);

            if(!SampleKernel(src, src_width, src_height, KernelDirectionHorizontal, f_x, f_y, type, h_ratio, v_ratio,
                             output))
            {
                if(errors)
                {
                    *errors = "Failure during horizontal resample operation.";
                }
                return false;
            }
        }

    for(uint32_t j = 0; j < dst_height; j++)
        for(uint32_t i = 0; i < dst_width; i++)
        {
            uint8_t* output = BLOCK_OFFSET_RGB24(dst, dst_width, i, j);

            float f_x = float(i);
            float f_y = float(j) * v_ratio;

            if(!SampleKernel(buffer.get(), dst_width, src_height, KernelDirectionVertical, f_x, f_y, type, h_ratio,
                             v_ratio, output))
            {
                if(errors)
                {
                    *errors = "Failure during vertical resample operation.";
                }
                return false;
            }
        }

    return true;
}

} // namespace rs