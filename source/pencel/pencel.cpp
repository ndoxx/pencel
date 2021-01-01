/* Take an input image and pixelize it using a
 * set of predefined colors.
 * For each pixel, the closest color in the set
 * to the actual input pixel value will be chosen.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <kibble/argparse/argparse.h>
#include <kibble/logger/dispatcher.h>
#include <kibble/logger/logger.h>
#include <kibble/logger/sink.h>
#include <kibble/math/color.h>

#include "lodepng/lodepng.h"

using namespace kb;
namespace fs = std::filesystem;

void init_logger()
{
    KLOGGER_START();
    KLOGGER(create_channel("pencel", 3));
    KLOGGER(attach_all("console_sink", std::make_unique<klog::ConsoleSink>()));
    KLOGGER(set_backtrace_on_error(false));
}

void show_error_and_die(ap::ArgParse& parser)
{
    for(const auto& msg : parser.get_errors())
        KLOGW("pencel") << msg << std::endl;

    KLOG("pencel", 1) << parser.usage() << std::endl;
    exit(0);
}

struct PencilInfo
{
    math::argb32_t heavy_trace;
    math::argb32_t light_trace;
    std::string name;
};

struct ColorMatchResult
{
    size_t index = 0;
    bool heavy = true;
    float distance = std::numeric_limits<float>::infinity();
};

struct Image
{
    std::vector<unsigned char> pixels;
    unsigned width, height;
};

ColorMatchResult best_match(math::argb32_t color, const std::vector<PencilInfo>& palette)
{
    ColorMatchResult result;
    for(size_t ii = 0; ii < palette.size(); ++ii)
    {
        const auto& info = palette[ii];
        // float dh = math::delta_E_cmetric(color, info.heavy_trace);
        // float dl = math::delta_E_cmetric(color, info.light_trace);
        float dh = math::delta_E2_CIE76(color, info.heavy_trace);
        float dl = math::delta_E2_CIE76(color, info.light_trace);
        // float dh = math::delta_E2_CIE94(color, info.heavy_trace);
        // float dl = math::delta_E2_CIE94(color, info.light_trace);
        if(dh < result.distance)
            result = {ii, true, dh};
        if(dl < result.distance)
            result = {ii, false, dl};
    }
    return result;
}

Image decode_png_file(const std::string& filename)
{
    Image image;
    unsigned error = lodepng::decode(image.pixels, image.width, image.height, filename.c_str(), LCT_RGB);
    if(error)
    {
        KLOGE("pencel") << "[lodepng] decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    return image;
}

// * Following code is a partial port of Ramenhut's image resampler
// Original is under BSD-2-Clause licence
// https://github.com/ramenhut/single-header-image-resampler
#define BLOCK_OFFSET_RGB24(ptr, width, x, y) (ptr + (3 * width) * y + 3 * x)
namespace resampler
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
                     uint32_t dst_height, KernelType type, ::std::string* errors = nullptr)
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
} // namespace resampler

void display_raw(const Image& image)
{
    KLOGR("pencel") << std::endl;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            KLOGR("pencel") << KF_(value) << "\u2588\u2588";
        }
        KLOGR("pencel") << std::endl;
    }
}

void display_palette(const Image& image, const std::vector<PencilInfo>& palette)
{
    KLOGR("pencel") << std::endl;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            auto bm = best_match(value, palette);
            auto bmc = (bm.heavy) ? palette[bm.index].heavy_trace : palette[bm.index].light_trace;
            KLOGR("pencel") << KF_(bmc) << "\u2588\u2588";
        }
        KLOGR("pencel") << std::endl;
    }
}

math::argb32_t redistribute_rgb(float r, float g, float b)
{
    float m = std::max(r, std::max(g, b));
    if(m <= 255)
        return math::pack_ARGB(uint8_t(r), uint8_t(g), uint8_t(b));

    float total = r + g + b;
    if(total > 3 * 255)
        return math::pack_ARGB(255, 255, 255);

    float x = (float(3 * 255) - total) / (3 * m - total);
    float gray = 255 - x * m;
    uint8_t out_r = uint8_t(gray + x * r);
    uint8_t out_g = uint8_t(gray + x * g);
    uint8_t out_b = uint8_t(gray + x * b);
    return math::pack_ARGB(out_r, out_g, out_b);
}

int main(int argc, char** argv)
{
    init_logger();

    ap::ArgParse parser("pencel", "0.1");
    parser.set_log_output([](const std::string& str) { KLOG("pencel", 1) << str << std::endl; });
    parser.set_exit_on_special_command(true);
    const auto& raw = parser.add_flag('r', "raw", "Display exact image colors.");
    const auto& source = parser.add_positional<std::string>("FILE", "Input PNG image file.");
    const auto& outwidth = parser.add_variable<int>('x', "width", "Output width.", 32);
    const auto& outheight = parser.add_variable<int>('y', "height", "Output height.", 32);

    bool success = parser.parse(argc, argv);

    if(!success)
        show_error_and_die(parser);

    fs::path imagepath(source());
    if(!fs::exists(imagepath))
    {
        KLOGE("pencel") << "Source file does not exist:" << std::endl;
        KLOGI << KS_PATH_ << imagepath << std::endl;
        exit(0);
    }
    if(imagepath.extension().string().compare(".png"))
    {
        KLOGE("pencel") << "Source file is not a PNG file:" << std::endl;
        KLOGI << KS_PATH_ << imagepath << std::endl;
        exit(0);
    }

    // * Import palette
    KLOGN("pencel") << "Importing palette:" << std::endl;
    std::vector<PencilInfo> palette;
    std::ifstream ifs("../data/faber-castell.gpl");
    std::string line;
    while(std::getline(ifs, line))
    {
        std::stringstream linestream(line);
        PencilInfo info;
        int r, g, b;
        linestream >> info.name >> r >> g >> b;
        info.heavy_trace = math::pack_ARGB(uint8_t(r), uint8_t(g), uint8_t(b));
        info.light_trace = redistribute_rgb(float(info.heavy_trace.r())*1.5f, float(info.heavy_trace.g())*1.5f,
                                            float(info.heavy_trace.b())*1.5f);

        KLOGI << KF_(info.heavy_trace) << "HH " << KF_(info.light_trace) << "LL " << KC_ << info.name << std::endl;
        palette.push_back(std::move(info));
    }

    // * Load image and resize it
    unsigned width = unsigned(outwidth());
    unsigned height = unsigned(outheight());

    auto src = decode_png_file(imagepath);
    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(width * height * 3);
    resampler::ResampleImage24(src.pixels.data(), src.width, src.height, img.pixels.data(), img.width, img.height,
                               resampler::KernelTypeBilinear);

    // * Display image
    if(raw())
        display_raw(img);
    else
    {
        display_raw(img);
        display_palette(img, palette);
    }

    return 0;
}