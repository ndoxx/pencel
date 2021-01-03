#pragma once
#include <glm/glm.hpp>
#include <kibble/math/color.h>
#include <limits>
#include <string>
#include <vector>

struct PencilInfo
{
    kb::math::argb32_t value;
    std::string name;
};

struct ColorMatchResult
{
    size_t index = 0;
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

kb::math::argb32_t hsl_transform(kb::math::argb32_t input, const glm::vec2& factors = {1.f, 1.f});

ColorMatchResult best_match(kb::math::argb32_t color, const std::vector<PencilInfo>& palette,
                            DeltaE method = DeltaE::CIE76, const glm::vec2& factors = {1.f, 1.f});

std::vector<PencilInfo> import_palette(const std::string& filename);