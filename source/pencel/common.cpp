#include "common.h"
#include <fstream>
#include <kibble/logger/logger.h>
#include <sstream>

using namespace kb;

kb::math::argb32_t hsl_transform(kb::math::argb32_t input, const glm::vec2& factors)
{
    if(factors[0] == 1.f && factors[1] == 1.f)
        return input;
    kb::math::ColorHSLA hsl(input);
    hsl.s = std::clamp(hsl.s * factors[0], 0.f, 1.f);
    hsl.l = std::clamp(hsl.l * factors[1], 0.f, 1.f);
    return kb::math::pack_ARGB(kb::math::to_RGBA(hsl));
}

ColorMatchResult best_match(math::argb32_t color, const std::vector<PencilInfo>& palette, DeltaE method,
                            const glm::vec2& factors)
{
    ColorMatchResult result;
    for(size_t ii = 0; ii < palette.size(); ++ii)
    {
        const auto& info = palette[ii];
        float dist;
        switch(method)
        {
        case DeltaE::CMETRIC:
            dist = math::delta_E_cmetric(hsl_transform(color, factors), info.value);
            break;
        case DeltaE::CIE76:
            dist = math::delta_E2_CIE76(hsl_transform(color, factors), info.value);
            break;
        case DeltaE::CIE94:
            dist = math::delta_E2_CIE94(hsl_transform(color, factors), info.value);
            break;
        }
        if(dist < result.distance)
            result = {ii, dist};
    }
    return result;
}

std::vector<PencilInfo> import_palette(const std::string& filename)
{
    KLOGN("pencel") << "Importing palette:" << std::endl;
    std::vector<PencilInfo> palette;
    std::ifstream ifs(filename);
    std::string line;
    while(std::getline(ifs, line))
    {
        std::stringstream linestream(line);
        PencilInfo info_heavy;
        PencilInfo info_light;
        std::string s_heavy, s_light;
        linestream >> info_heavy.name >> s_heavy >> s_light;
        info_light.name = info_heavy.name + '*';
        char* p;
        uint32_t heavy = uint32_t(std::strtol(s_heavy.c_str(), &p, 16));
        if(*p == 0)
        {
            info_heavy.value = {heavy};
        }
        uint32_t light = uint32_t(std::strtol(s_light.c_str(), &p, 16));
        if(*p == 0)
        {
            info_light.value = {light};
        }
        // KLOGI << KF_(info.heavy_trace) << "HH " << KF_(info.light_trace) << "LL " << KC_ << info.name << std::endl;
        KLOG("pencel", 1) << KF_(info_heavy.value) << "\u2588\u2588" << KF_(info_light.value) << "\u2588\u2588" << KC_
                          << ' ';
        palette.push_back(std::move(info_heavy));
        palette.push_back(std::move(info_light));
    }

    KLOG("pencel", 1) << std::endl;
    return palette;
}