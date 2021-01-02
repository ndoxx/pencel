#pragma once

#include "common.h"
#include <glm/glm.hpp>
#include <vector>

enum class UpdateMethod
{
	FDSA,
	SPSA
};

struct DescentParameters
{
    glm::vec2 initial_control = {1.f, 1.f};
    float initial_step = 1.f;
    float initial_epsilon = 0.5f;
    float convergence_delta = 0.0005f;
    float alpha = 0.602f;
    float gamma = 0.101f;
    size_t max_iter = 200;
    UpdateMethod method = UpdateMethod::SPSA;
};

class HSLOptimizer
{
public:
    glm::vec2 optimize_gd(const Image& image, const std::vector<PencilInfo>& palette, const DescentParameters& params);

private:
    glm::vec3 loss(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& control);
    float combine(glm::vec3 loss_vector);

private:
	glm::vec3 initial_loss_;
};