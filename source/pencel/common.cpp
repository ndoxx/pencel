#include "common.h"

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