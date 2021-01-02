#include "optimizer.h"
#include "resampler.h"
#include <fstream>
#include <kibble/logger/logger.h>
#include <random>
#include <set>

using namespace kb;

struct OptimizationStep
{
    size_t colors_used = 0;
    float distance = 0.f;
    float l_factor = 1.f;
    float s_factor = 1.f;
};

inline void constrain(glm::vec2& uu)
{
    float uu0_clamp = std::max(std::min(uu[0], 2.f), 0.f);
    float uu1_clamp = std::max(std::min(uu[1], 4.f), 0.f);
    uu = {uu0_clamp, uu1_clamp};
}

inline float bernoulli_remap(bool value) { return value ? 1.f : -1.f; }

inline void exponential_moving_average(float& accumulator, float new_value, float alpha)
{
    accumulator = (alpha * new_value) + (1.f - alpha) * accumulator;
}

glm::vec2 HSLOptimizer::optimize_gd(const Image& image, const std::vector<PencilInfo>& palette,
                                    const DescentParameters& params)
{
    initial_loss_ = {1.f, 1.f, 1.f};
    initial_loss_ = loss(image, palette, params.initial_control);

    std::random_device r;
    std::default_random_engine gen(r());
    std::bernoulli_distribution dis(0.5);

    size_t iter = 0;
    float filtered_loss = 1.f;
    float old_loss = std::numeric_limits<float>::infinity();
    glm::vec2 uu = params.initial_control;
    std::array<glm::vec2, 2> dir = {glm::vec2{1.f, 0.f}, glm::vec2{0.f, 1.f}};
    while(iter < params.max_iter && std::abs(filtered_loss - old_loss) > params.convergence_delta)
    {
        float ak = params.initial_step / std::pow(float(iter + 1), params.alpha);
        float ck = params.initial_epsilon / std::pow(float(iter + 1), params.gamma);
        float current_loss = 0.f;

        glm::vec2 g;
        switch(params.method)
        {
        case UpdateMethod::FDSA: {
            float forward_loss_0 = combine(loss(image, palette, uu + ck * dir[0]));
            float backward_loss_0 = combine(loss(image, palette, uu - ck * dir[0]));
            float forward_loss_1 = combine(loss(image, palette, uu + ck * dir[1]));
            float backward_loss_1 = combine(loss(image, palette, uu - ck * dir[1]));
            float g0 = (0.5f / ck) * (forward_loss_0 - backward_loss_1);
            float g1 = (0.5f / ck) * (forward_loss_1 - backward_loss_1);
            g = {g0, g1};
            current_loss = 0.25f * (forward_loss_0 + backward_loss_0 + forward_loss_1 + backward_loss_1);
            break;
        }
        case UpdateMethod::SPSA: {
            glm::vec2 rvec{bernoulli_remap(dis(gen)), bernoulli_remap(dis(gen))};
            rvec = glm::normalize(rvec);
            float forward_loss = combine(loss(image, palette, uu + ck * rvec));
            float backward_loss = combine(loss(image, palette, uu - ck * rvec));
            float h = (forward_loss - backward_loss);
            g = {h * (0.5f / (ck * rvec[0])), h * (0.5f / (ck * rvec[1]))};
            current_loss = 0.5f * (forward_loss + backward_loss);
            break;
        }
        }

        // Update and constrain control parameters
        uu -= ak * g;
        constrain(uu);

        // IIR filter applied to the current loss to limit sensitivity to loss jittering
        old_loss = filtered_loss;
        exponential_moving_average(filtered_loss, current_loss, 0.1f);

        KLOG("pencel", 1) << "\033[1A\033[K\033[1A\033[K\033[1A\033[K\033[1A\033[K" << std::endl;
        KLOG("pencel", 1) << "Iteration: " << iter << std::endl;
        KLOGI << "ck: " << ck << " ak: " << ak << " loss: " << current_loss
              << " delta: " << std::abs(filtered_loss - old_loss) << std::endl;
        KLOGI << "control: [" << uu[0] << ',' << uu[1] << ']' << std::endl;

        ++iter;
    }
    return uu;
}

inline float euclidean_dist(float a, float b) { return std::sqrt((a - b) * (a - b)); }

glm::vec3 HSLOptimizer::loss(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& control)
{
    auto argb_cmp = [](math::argb32_t a, math::argb32_t b) { return a.value < b.value; };

    // Low overall perceptive color difference w.r.t original image is penalized
    float fidelity_loss = 0.f;
    std::set<math::argb32_t, decltype(argb_cmp)> colors_used;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            // The fidelity loss is computed using the faster CMetric color difference
            auto bm = best_match(value, palette, DeltaE::CMETRIC, control);
            fidelity_loss += bm.distance;
            if(bm.heavy)
                colors_used.insert(palette[bm.index].heavy_trace);
            else
                colors_used.insert(palette[bm.index].light_trace);
        }
    }

    // Low color diversity is penalized
    float diversity_loss = 1.f - float(colors_used.size()) / float(2 * palette.size());
    // Heavy HSL manipulation is penalized
    float transform_penalty = 0.6f * euclidean_dist(control[0], 1.f) + 0.4f * euclidean_dist(control[1], 1.f);
    // Normalize loss using initial loss as a reference
    glm::vec3 loss = {fidelity_loss / initial_loss_[0], diversity_loss / initial_loss_[1], transform_penalty};
    // KLOG("pencel",1) << "loss: " << loss[0] << '/' << loss[1] << '/' << loss[2] << std::endl;
    return loss;
}

float HSLOptimizer::combine(glm::vec3 loss_vector)
{
    // return (loss_vector[0] + loss_vector[1] + loss_vector[2]) / 3.f;
    return glm::length(loss_vector);
}
