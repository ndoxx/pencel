#include "common.h"

using namespace kb;

ColorMatchResult best_match(math::argb32_t color, const std::vector<PencilInfo>& palette, DeltaE method, const glm::vec2& factors)
{
    ColorMatchResult result;
    for(size_t ii = 0; ii < palette.size(); ++ii)
    {
        const auto& info = palette[ii];
        float dh, dl;
        switch(method)
        {
        case DeltaE::CMETRIC:
            dh = math::delta_E_cmetric(hsl_transform(color, factors), info.heavy_trace);
            dl = math::delta_E_cmetric(hsl_transform(color, factors), info.light_trace);
            break;
        case DeltaE::CIE76:
            dh = math::delta_E2_CIE76(hsl_transform(color, factors), info.heavy_trace);
            dl = math::delta_E2_CIE76(hsl_transform(color, factors), info.light_trace);
            break;
        case DeltaE::CIE94:
            dh = math::delta_E2_CIE94(hsl_transform(color, factors), info.heavy_trace);
            dl = math::delta_E2_CIE94(hsl_transform(color, factors), info.light_trace);
            break;
        }
        if(dh < result.distance)
            result = {ii, true, dh};
        if(dl < result.distance)
            result = {ii, false, dl};
    }
    return result;
}