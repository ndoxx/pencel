#pragma once

#include "common.h"
#include <glm/glm.hpp>
#include <vector>

enum class Method
{
	FDSA,
	SPSA
};

struct DescentParameters
{
    glm::vec2 initial_control = {0.5f, 0.5f};
    float initial_step = 1.f;
    float initial_epsilon = 0.2f;
    float convergence_delta = 0.01f;
    size_t max_iter = 100;
    Method method = Method::FDSA;
};

class HSLOptimizer
{
public:
    glm::vec2 optimize_exhaustive(const Image& image, const std::vector<PencilInfo>& palette);
    glm::vec2 optimize_gd(const Image& image, const std::vector<PencilInfo>& palette, const DescentParameters& params);

private:
    glm::vec3 loss(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& control);
    float combine(glm::vec3 loss_vector);
};