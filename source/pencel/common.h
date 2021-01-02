#pragma once
#include <kibble/math/color.h>
#include <glm/glm.hpp>
#include <limits>
#include <string>
#include <vector>

struct PencilInfo
{
    kb::math::argb32_t heavy_trace;
    kb::math::argb32_t light_trace;
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

enum class DeltaE
{
    CMETRIC,
    CIE76,
    CIE94
};

inline kb::math::argb32_t hsl_transform(kb::math::argb32_t input, const glm::vec2& factors = {1.f,1.f})
{
    if(factors[0] == 1.f && factors[1] == 1.f)
        return input;
    kb::math::ColorHSLA hsl(input);
    hsl.l = std::clamp(hsl.l * factors[0], 0.f, 1.f);
    hsl.s = std::clamp(hsl.s * factors[1], 0.f, 1.f);
    return kb::math::pack_ARGB(kb::math::to_RGBA(hsl));
}

ColorMatchResult best_match(kb::math::argb32_t color, const std::vector<PencilInfo>& palette, DeltaE method = DeltaE::CIE76,
                            const glm::vec2& factors = {1.f,1.f});