
#include "common.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <kibble/argparse/argparse.h>
#include <kibble/hash/hash.h>
#include <kibble/logger/dispatcher.h>
#include <kibble/logger/logger.h>
#include <kibble/logger/sink.h>
#include <map>

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

int main(int argc, char** argv)
{
    init_logger();

    ap::ArgParse parser("pencel", "0.1");
    parser.set_log_output([](const std::string& str) { KLOG("pencel", 1) << str << std::endl; });
    parser.set_exit_on_special_command(true);
    const auto& source = parser.add_positional<std::string>("INPUT_FILE", "Input grid file.");
    bool success = parser.parse(argc, argv);

    if(!success)
        show_error_and_die(parser);

    fs::path gridpath(source());
    if(!fs::exists(gridpath))
    {
        KLOGE("pencel") << "Source file does not exist:" << std::endl;
        KLOGI << KS_PATH_ << gridpath << std::endl;
        return 0;
    }

    // * Read palette
    auto palette = import_palette("../data/faber-castell.txt");

    // * Read grid file
    size_t width;
    size_t height;
    std::vector<std::string> grid;
    std::ifstream ifs(gridpath);
    ifs >> width >> height;

    if(width * height > 512 * 512)
    {
        KLOGE("pencel") << "Grid is too big." << std::endl;
        return 0;
    }
    KLOG("pencel", 1) << "Dimensions: " << width << 'x' << height << std::endl;

    grid.reserve(width * height);

    std::map<hash_t, size_t> color_table;
    size_t max_name_size = 0;
    while(!ifs.eof())
    {
        std::string color;
        ifs >> color;
        if(color.size())
        {
            grid.push_back(color);

            hash_t hname = H_(color);
            if(color_table.find(hname) == color_table.end())
            {
                size_t idx = 0;
                for(; idx < palette.size(); ++idx)
                    if(H_(palette[idx].name) == hname)
                        break;
                color_table.insert({hname, idx});
            }

            if(color.size() > max_name_size)
                max_name_size = color.size();
        }
    }

    // * Display colors needed
    KLOG("pencel", 1) << "Colors needed for this artwork:" << std::endl;
    for(auto&& [hname, idx] : color_table)
    {
        KLOGR("pencel") << KF_(palette[idx].value) << "\u2588\u2588 " << KC_ << std::left
                        << std::setw(int(max_name_size)) << palette[idx].name << ' ';
    }
    KLOGR("pencel") << std::endl;

    // * Display grid sector
    size_t sector_w = 8;
    size_t sector_h = 8;
    size_t max_sector = (width / sector_w) * (height / sector_h) - 1;
    size_t sector = 0;

    bool loop = true;

    while(loop)
    {
        size_t sector_col = sector % (width / sector_w);
        size_t sector_row = sector / (height / sector_h);
        size_t start_col = sector_w * sector_col;
        size_t start_row = sector_h * sector_row;

        KLOGR("pencel") << KC_ << "----------------------------------" << std::endl;
        KLOGR("pencel") << "Sector: " << sector << " (" << sector_row << ',' << sector_col << ')' << std::endl;
        for(size_t jj = 0; jj < sector_h; ++jj)
        {
            size_t yy = start_row + jj;
            // Show grid sector with colors
            for(size_t ii = 0; ii < sector_w; ++ii)
            {
                size_t xx = start_col + ii;
                size_t idx = yy * width + xx;

                hash_t hname = H_(grid[idx]);
                size_t color_idx = color_table.at(hname);
                KLOGR("pencel") << KF_(palette.at(color_idx).value) << "\u2588\u2588";
            }
            // Show coloring instructions
            KLOGR("pencel") << KC_ << "    ";
            for(size_t ii = 0; ii < sector_w; ++ii)
            {
                size_t xx = start_col + ii;
                size_t idx = yy * width + xx;
                KLOGR("pencel") << std::left << std::setw(int(max_name_size)) << grid[idx] << ' ';
            }

            KLOGR("pencel") << std::endl;
        }

        KLOGR("pencel") << "Next sector: " << std::endl;
        std::cin >> sector;
        if(sector > max_sector)
            sector = max_sector;
    }

    return 0;
}