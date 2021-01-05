#pragma once

#include "common.h"
#include <glm/glm.hpp>
#include <vector>
#include <kibble/algorithm/optimization.h>

class HSLOptimizer
{
public:
    // Optimize HSL transformation on input image so as to maximize color diversity
    // Gradient descent: Finite Difference Stochastic Approximation (slow)
    glm::vec2 optimize_fdsa(const Image& image, const std::vector<PencilInfo>& palette,
                            const kb::opt::DescentParameters<glm::vec2>& params);
    // Gradient descent: Simultaneous Perturbation Stochastic Approximation (fast)
    glm::vec2 optimize_spsa(const Image& image, const std::vector<PencilInfo>& palette,
                            const kb::opt::DescentParameters<glm::vec2>& params);

    void sample_loss_manifold(const std::string& filename, const Image& image, const std::vector<PencilInfo>& palette,
                              size_t size_x, size_t size_y);

private:
    // Loss functions: results are collected in a vector
    glm::vec3 loss(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& control);
    // Combine the different loss components into a single scalar
    float scalarize(glm::vec3 loss_vector);

private:
    glm::vec3 initial_loss_;
};