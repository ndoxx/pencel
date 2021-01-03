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
    float uu0_clamp = std::max(std::min(uu[0], 4.f), 0.f);
    float uu1_clamp = std::max(std::min(uu[1], 2.f), 0.f);
    uu = {uu0_clamp, uu1_clamp};
}

inline float bernoulli_remap(bool value) { return value ? 1.f : -1.f; }

inline void exponential_moving_average(float& accumulator, float new_value, float alpha)
{
    accumulator = (alpha * new_value) + (1.f - alpha) * accumulator;
}

glm::vec2 HSLOptimizer::optimize_fdsa(const Image& image, const std::vector<PencilInfo>& palette,
                                      const DescentParameters& params)
{
    initial_loss_ = {1.f, 1.f, 1.f};
    initial_loss_ = loss(image, palette, params.initial_control);

    size_t iter = 0;
    float filtered_loss = 1.f;
    float old_loss = std::numeric_limits<float>::infinity();
    glm::vec2 uu = params.initial_control;
    std::array<glm::vec2, 2> dir = {glm::vec2{1.f, 0.f}, glm::vec2{0.f, 1.f}};
    while(iter < params.max_iter && std::abs(filtered_loss - old_loss) > params.convergence_delta)
    {
        float ak = params.initial_step / std::pow(float(iter + 1) + params.learning_bias, params.alpha);
        float ck = params.initial_epsilon / std::pow(float(iter + 1), params.gamma);

        // Compute forward and backward losses for each dimension of the control vector
        float forward_loss_0 = scalarize(loss(image, palette, uu + ck * dir[0]));
        float backward_loss_0 = scalarize(loss(image, palette, uu - ck * dir[0]));
        float forward_loss_1 = scalarize(loss(image, palette, uu + ck * dir[1]));
        float backward_loss_1 = scalarize(loss(image, palette, uu - ck * dir[1]));
        float g0 = (0.5f / ck) * (forward_loss_0 - backward_loss_1);
        float g1 = (0.5f / ck) * (forward_loss_1 - backward_loss_1);

        // Update and constrain control parameters
        uu -= ak * glm::vec2{g0, g1};
        constrain(uu);

        // IIR filter applied to the current loss to limit sensitivity to loss jittering
        float current_loss = 0.25f * (forward_loss_0 + backward_loss_0 + forward_loss_1 + backward_loss_1);
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

glm::vec2 HSLOptimizer::optimize_spsa(const Image& image, const std::vector<PencilInfo>& palette,
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
    while(iter < params.max_iter && std::abs(filtered_loss - old_loss) > params.convergence_delta)
    {
        float ak = params.initial_step / std::pow(float(iter + 1) + params.learning_bias, params.alpha);
        float ck = params.initial_epsilon / std::pow(float(iter + 1), params.gamma);

        // Compute forward and backward loss given a random perturbation
        glm::vec2 del = glm::normalize(glm::vec2{bernoulli_remap(dis(gen)), bernoulli_remap(dis(gen))});
        float forward_loss = scalarize(loss(image, palette, uu + ck * del));
        float backward_loss = scalarize(loss(image, palette, uu - ck * del));
        float h = (forward_loss - backward_loss);
        glm::vec2 g = {h * (0.5f / (ck * del[0])), h * (0.5f / (ck * del[1]))};

        // Update and constrain control parameters
        uu -= ak * g;
        constrain(uu);

        // IIR filter applied to the current loss to limit sensitivity to loss jittering
        float current_loss = 0.5f * (forward_loss + backward_loss);
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

void HSLOptimizer::sample_loss_manifold(const std::string& filename, const Image& image,
                                        const std::vector<PencilInfo>& palette, size_t size_x, size_t size_y)
{
    initial_loss_ = {1.f, 1.f, 1.f};
    initial_loss_ = loss(image, palette, {1.f, 1.f});

    KLOGN("pencel") << "Sampling loss manifold." << std::endl;
    std::ofstream ofs(filename);
    for(size_t ii = 0; ii < size_x; ++ii)
    {
        float ss = 4.f * float(ii) / float(size_x - 1);
        for(size_t jj = 0; jj < size_y; ++jj)
        {
            float ll = 2.f * float(jj) / float(size_y - 1);
            float J = scalarize(loss(image, palette, {ss, ll}));
            ofs << ss << ' ' << ll << ' ' << J << std::endl;
        }
        ofs << std::endl;
        KLOGI << size_t(std::round(100.f * float(ii) / float(size_x - 1))) << '%' << std::endl;
    }
}

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
            colors_used.insert(palette[bm.index].value);
        }
    }

    // Low color diversity is penalized
    float diversity_loss = 1.f - float(colors_used.size()) / float(2 * palette.size());
    // Heavy HSL manipulation is penalized
    float s_factor = (control[0] - 1.f) * (control[0] - 1.f);
    float l_factor = (control[1] - 1.f) * (control[1] - 1.f);
    float transform_penalty = 0.2f * s_factor + 0.8f * l_factor;
    // Normalize loss using initial loss as a reference
    glm::vec3 loss = {fidelity_loss / initial_loss_[0], diversity_loss / initial_loss_[1], transform_penalty};
    return loss;
}

float HSLOptimizer::scalarize(glm::vec3 loss_vector)
{
    return (loss_vector[0] + loss_vector[1] + loss_vector[2]) / 3.f;
}
