#include "optimizer.h"
#include "resampler.h"
#include <fstream>
#include <kibble/logger/logger.h>
#include <random>
#include <set>


using namespace kb;

namespace kb::opt
{
template <> struct ControlTraits<glm::vec2>
{
	using size_type = int;
    static constexpr size_type size() { return 2; }
    static void normalize(glm::vec2& vec) { vec = glm::normalize(vec); }
};
}

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
                                      const kb::opt::DescentParameters<glm::vec2>& params)
{
    initial_loss_ = {1.f, 1.f, 1.f};
    initial_loss_ = loss(image, palette, params.initial_control);

    opt::StochasticDescentOptimizer<glm::vec2> optimizer;
    optimizer.set_loss([this,&image,&palette](const glm::vec2& control){return scalarize(loss(image, palette, control));});
    optimizer.set_constraint([](glm::vec2& control){constrain(control);});
    optimizer.set_iteration_callback([](size_t iter, const glm::vec2& control, float loss)
    {
        KLOG("pencel", 1) << "Iteration: " << iter << " control: [" << control[0] << ',' << control[1] << "] loss=" << loss << std::endl;
    });

    return optimizer.FDSA(params);

}

glm::vec2 HSLOptimizer::optimize_spsa(const Image& image, const std::vector<PencilInfo>& palette,
                                      const kb::opt::DescentParameters<glm::vec2>& params)
{
    initial_loss_ = {1.f, 1.f, 1.f};
    initial_loss_ = loss(image, palette, params.initial_control);

    opt::StochasticDescentOptimizer<glm::vec2> optimizer;
    optimizer.set_loss([this,&image,&palette](const glm::vec2& control){return scalarize(loss(image, palette, control));});
    optimizer.set_constraint([](glm::vec2& control){constrain(control);});
    optimizer.set_iteration_callback([](size_t iter, const glm::vec2& control, float loss)
    {
        KLOG("pencel", 1) << "Iteration: " << iter << " control: [" << control[0] << ',' << control[1] << "] loss=" << loss << std::endl;
    });

    return optimizer.SPSA(params);
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
