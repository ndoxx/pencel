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
#include <vector>

#include <kibble/argparse/argparse.h>
#include <kibble/logger/dispatcher.h>
#include <kibble/logger/logger.h>
#include <kibble/logger/sink.h>

#include "lodepng/lodepng.h"
#include "resampler.h"
#include "optimizer.h"
#include "common.h"

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

void display_palette(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& factors)
{
    KLOGR("pencel") << std::endl;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            auto bm = best_match(value, palette, DeltaE::CIE94, factors);
            auto bmc = (bm.heavy) ? palette[bm.index].heavy_trace : palette[bm.index].light_trace;
            KLOGR("pencel") << KF_(bmc) << "\u2588\u2588";
        }
        KLOGR("pencel") << std::endl;
    }
}

int main(int argc, char** argv)
{
    init_logger();

    ap::ArgParse parser("pencel", "0.1");
    parser.set_log_output([](const std::string& str) { KLOG("pencel", 1) << str << std::endl; });
    parser.set_exit_on_special_command(true);
    const auto& raw = parser.add_flag('r', "raw", "Display exact image colors.");
    const auto& optimize = parser.add_flag(
        'o', "optimize",
        "Allow original picture lightness and saturation to vary, which could produce better looking results.");
    const auto& source = parser.add_positional<std::string>("FILE", "Input PNG image file.");
    const auto& outwidth = parser.add_variable<int>('x', "width", "Output width.", 32);
    const auto& outheight = parser.add_variable<int>('y', "height", "Output height.", 32);
    const auto& saturation_factor = parser.add_variable<float>('s', "saturation", "Saturation factor.", 1.f);
    const auto& lightness_factor = parser.add_variable<float>('l', "lightness", "Lightness factor.", 1.f);

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
        // KLOGI << KF_(info.heavy_trace) << "HH " << KF_(info.light_trace) << "LL " << KC_ << info.name << std::endl;
        KLOG("pencel",1) << KF_(info.heavy_trace) << "\u2588\u2588" << KF_(info.light_trace) << "\u2588\u2588" << KC_ << ' ';
        palette.push_back(std::move(info));
    }

    KLOG("pencel",1) << std::endl;

    // * Load image and resize it
    unsigned width = unsigned(outwidth());
    unsigned height = unsigned(outheight());

    auto src = decode_png_file(imagepath);
    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(width * height * 3);
    rs::ResampleImage24(src.pixels.data(), src.width, src.height, img.pixels.data(), img.width, img.height,
                        rs::KernelTypeLanczos5);

    if(raw())
    {
        display_raw(img);
        return 0;
    }

    // * Optimize lightness
    glm::vec2 factors{lightness_factor(), saturation_factor()};
    if(optimize())
    {
        Image img2;
        img2.width = img.width / 2;
        img2.height = img.height / 2;
        img2.pixels.resize(width * height * 3);
        rs::ResampleImage24(src.pixels.data(), src.width, src.height, img2.pixels.data(), img2.width, img2.height,
                            rs::KernelTypeLanczos5);

        HSLOptimizer optimizer;
        DescentParameters params;
        params.method = UpdateMethod::FDSA;
        // params.method = UpdateMethod::SPSA;
        params.initial_control = factors;
        params.initial_step = 1.f;
        params.initial_epsilon = 0.5f;
        params.convergence_delta = 0.0005f;
        params.alpha = 0.602f;
        params.gamma = 0.101f;
        params.max_iter = 200;

        factors = optimizer.optimize_gd(img2, palette, params);
    }

    // * Display image
    // display_raw(img);
    display_palette(img, palette, factors);

    return 0;
}