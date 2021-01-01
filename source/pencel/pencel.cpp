/* Take an input image and pixelize it using a
 * set of predefined colors.
 * For each pixel, the closest color in the set
 * to the actual input pixel value will be chosen.
 */

#include <cmath>
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
#include "resampler.h"

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

struct OptimizationRound
{
    size_t colors_used = 0;
    float distance = 0.f;
    float factor = 1.f;
};

struct Image
{
    std::vector<unsigned char> pixels;
    unsigned width, height;
};

inline math::argb32_t lighten(math::argb32_t input, float factor)
{
    if(factor == 1.f)
        return input;
    math::ColorHSLA hsl(input);
    hsl.l = std::clamp(hsl.l * factor, 0.f, 1.f);
    return math::pack_ARGB(math::to_RGBA(hsl));
}

ColorMatchResult best_match(math::argb32_t color, const std::vector<PencilInfo>& palette, float lfactor = 1.f)
{
    ColorMatchResult result;
    for(size_t ii = 0; ii < palette.size(); ++ii)
    {
        const auto& info = palette[ii];
        // float dh = math::delta_E_cmetric(color, info.heavy_trace);
        // float dl = math::delta_E_cmetric(color, info.light_trace);
        float dh = math::delta_E2_CIE76(lighten(color, lfactor), info.heavy_trace);
        float dl = math::delta_E2_CIE76(lighten(color, lfactor), info.light_trace);
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

void display_palette(const Image& image, const std::vector<PencilInfo>& palette, float lightness_factor = 1.f)
{
    KLOGR("pencel") << std::endl;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            auto bm = best_match(value, palette, lightness_factor);
            auto bmc = (bm.heavy) ? palette[bm.index].heavy_trace : palette[bm.index].light_trace;
            KLOGR("pencel") << KF_(bmc) << "\u2588\u2588";
        }
        KLOGR("pencel") << std::endl;
    }
}

float optimize_lightness(const Image& image, const std::vector<PencilInfo>& palette)
{
    // Find the lightness factor that, when applied to the original image, will
    // simultaneously maximize the color diversity and minimize the overall
    // perceptive distance with colors in the palette

    size_t nsteps = 20;
    std::vector<OptimizationRound> rounds;
    auto cmp = [](math::argb32_t a, math::argb32_t b) { return a.value < b.value; };
    float max_dist = 0.f;
    for(size_t ii = 0; ii < nsteps; ++ii)
    {
        float factor = 0.7f + 0.6f * float(ii) / float(nsteps - 1);
        std::set<math::argb32_t, decltype(cmp)> colors_used;
        float dist = 0.f;
        for(unsigned int row = 0; row < image.height; ++row)
        {
            for(unsigned int col = 0; col < image.width; ++col)
            {
                auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
                auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
                auto bm = best_match(value, palette, factor);
                dist += bm.distance * bm.distance;
                if(bm.heavy)
                    colors_used.insert(palette[bm.index].heavy_trace);
                else
                    colors_used.insert(palette[bm.index].light_trace);
            }
        }
        dist = std::sqrt(dist);
        rounds.push_back({colors_used.size(), dist, factor});
        if(dist > max_dist)
            max_dist = dist;

        KLOG("pencel", 1) << "Optimizing lightness: " << size_t(std::round(100.f * float(ii) / float(nsteps - 1)))
                          << '%' << std::endl;
    }

    float best_pareto = std::numeric_limits<float>::infinity();
    float best_factor = 1.f;
    for(const auto& oround : rounds)
    {
        float dist_score = oround.distance / max_dist;
        float diversity_score = float(2 * palette.size()) / float(oround.colors_used);
        float pareto = std::sqrt(diversity_score * diversity_score + dist_score * dist_score);
        KLOG("pencel", 1) << "factor: " << oround.factor << " div: " << diversity_score << " dist:" << dist_score
                          << " pareto: " << pareto << std::endl;
        if(pareto < best_pareto)
        {
            best_pareto = pareto;
            best_factor = oround.factor;
        }
    }
    KLOG("pencel", 1) << "Best score: " << best_pareto << " -> factor: " << best_factor << std::endl;

    return best_factor;
}

int main(int argc, char** argv)
{
    init_logger();

    ap::ArgParse parser("pencel", "0.1");
    parser.set_log_output([](const std::string& str) { KLOG("pencel", 1) << str << std::endl; });
    parser.set_exit_on_special_command(true);
    const auto& raw = parser.add_flag('r', "raw", "Display exact image colors.");
    const auto& opt_lightness =
        parser.add_flag('l', "optimize-lightness",
                        "Allow original picture lightness to vary, which could produce better looking results.");
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
    std::ifstream ifs("../data/faber-castell.txt");
    std::string line;
    while(std::getline(ifs, line))
    {
        std::stringstream linestream(line);
        PencilInfo info;
        std::string s_heavy, s_light;
        linestream >> info.name >> s_heavy >> s_light;
        char* p;
        uint32_t heavy = uint32_t(std::strtol(s_heavy.c_str(), &p, 16));
        if(*p == 0)
            info.heavy_trace = {heavy};
        uint32_t light = uint32_t(std::strtol(s_light.c_str(), &p, 16));
        if(*p == 0)
            info.light_trace = {light};
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
    rs::ResampleImage24(src.pixels.data(), src.width, src.height, img.pixels.data(), img.width, img.height,
                        rs::KernelTypeBilinear);

    // * Optimize lightness
    float best_factor = 1.f;
    if(opt_lightness())
    {
        Image img2;
        img2.width = 32;
        img2.height = 32;
        img2.pixels.resize(width * height * 3);
        rs::ResampleImage24(src.pixels.data(), src.width, src.height, img2.pixels.data(), img2.width, img2.height,
                            rs::KernelTypeBilinear);

        best_factor = optimize_lightness(img2, palette);
    }

    // * Display image
    if(raw())
        display_raw(img);
    else
    {
        display_raw(img);
        display_palette(img, palette, best_factor);
    }

    return 0;
}