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

inline float calculate_transform_penalty(float factor)
{
    // This polynomial evaluates to 1 at 0.5, 0 at 1 and 1 at 2
    // p(x) = -2*x^3 + 9*x^2 - 12*x + 5
    return 5.f - 12 * factor + 9 * factor * factor - 2 * factor * factor * factor;
}

inline float factor_sampling_density(float ss)
{
    // This polynomial plateaus at 1 around ss=0.5
    // It allows to sample the optimization manifold more densely for factor values close to 1
    // p(x) = 4.166666666666667*x^3 - 5.25*x^2 + 2.5833333333333335*x + 0.5
    return 0.5f + 2.5833333333333335f * ss - 5.25f * ss * ss + 4.166666666666667f * ss * ss * ss;
}

glm::vec2 HSLOptimizer::optimize_exhaustive(const Image& image, const std::vector<PencilInfo>& palette)
{
    // Find the lightness factor that, when applied to the original image, will
    // simultaneously maximize the color diversity and minimize the overall
    // perceptive distance with colors in the palette

    size_t nsteps = 25;
    std::vector<OptimizationStep> steps;
    auto argb_cmp = [](math::argb32_t a, math::argb32_t b) { return a.value < b.value; };
    float max_dist = 0.f;
    float min_dist = std::numeric_limits<float>::infinity();
    size_t max_used = 0;
    size_t min_used = std::numeric_limits<size_t>::max();
    for(size_t ii = 0; ii < nsteps; ++ii)
    {
        float xx = 0.25f + 0.5f * float(ii) / float(nsteps - 1);
        float l_factor = factor_sampling_density(xx);
        for(size_t jj = 0; jj < nsteps; ++jj)
        {
            float yy = float(jj) / float(nsteps - 1);
            float s_factor = factor_sampling_density(yy);
            std::set<math::argb32_t, decltype(argb_cmp)> colors_used;
            float dist = 0.f;
            for(unsigned int row = 0; row < image.height; ++row)
            {
                for(unsigned int col = 0; col < image.width; ++col)
                {
                    auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
                    auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
                    auto bm = best_match(value, palette, DeltaE::CIE94, {l_factor, s_factor});
                    // auto bm = best_match(value, palette, DeltaE::CMETRIC, l_factor, s_factor);
                    dist += bm.distance * bm.distance;
                    if(bm.heavy)
                        colors_used.insert(palette[bm.index].heavy_trace);
                    else
                        colors_used.insert(palette[bm.index].light_trace);
                }
            }
            dist = std::sqrt(dist);
            steps.push_back({colors_used.size(), dist, l_factor, s_factor});
            if(dist > max_dist)
                max_dist = dist;
            if(dist < min_dist)
                min_dist = dist;
            if(colors_used.size() > max_used)
                max_used = colors_used.size();
            if(colors_used.size() < min_used)
                min_used = colors_used.size();
        }
        KLOG("pencel", 1) << "Optimizing: " << size_t(std::round(100.f * float(ii) / float(nsteps - 1))) << '%'
                          << std::endl;
    }

    float best_pareto = std::numeric_limits<float>::infinity();
    float best_l_factor = 1.f;
    float best_s_factor = 1.f;
    std::ofstream ofs("manifold.dat");
    for(size_t ii = 0; ii < steps.size(); ++ii)
    {
        const auto& ostep = steps[ii];
        // Penalize extreme lightness factors
        float closeness_score =
            0.5f * (calculate_transform_penalty(ostep.l_factor) + calculate_transform_penalty(ostep.s_factor));
        // Penalize big overall perceptive difference w.r.t original
        float fidelity_score = (ostep.distance - min_dist) / (max_dist - min_dist);
        // Penalize low color diversity
        float diversity_score = 1.f - float(ostep.colors_used - min_used) / float(max_used - min_used);
        uint32_t color = (uint32_t(255.f * 0.5f * ostep.l_factor) << 16) | uint32_t(255.f * 0.5f * ostep.s_factor);
        ofs << closeness_score << ' ' << fidelity_score << ' ' << diversity_score << ' ' << color << std::endl;
        float pareto =
            diversity_score * diversity_score + fidelity_score * fidelity_score + closeness_score * closeness_score;
        KLOG("pencel", 1) << "lmod: " << ostep.l_factor << " smod: " << ostep.s_factor << " div: " << diversity_score
                          << " dis: " << fidelity_score << " clo: " << closeness_score << " pareto: " << pareto
                          << std::endl;
        if(pareto < best_pareto)
        {
            best_pareto = pareto;
            best_l_factor = ostep.l_factor;
            best_s_factor = ostep.s_factor;
        }
    }
    KLOG("pencel", 1) << "Best score: " << best_pareto << " -> l_factor: " << best_l_factor
                      << " s_factor: " << best_s_factor << std::endl;

    return {best_l_factor, best_s_factor};
}

glm::vec2 HSLOptimizer::optimize_gd(const Image& image, const std::vector<PencilInfo>& palette,
                                    const DescentParameters& params)
{
    std::random_device r;
    std::default_random_engine gen(r());
    std::uniform_real_distribution<float> dis(-1.f, 1.f);

    size_t iter = 0;
    float ste = params.initial_step;
    float eps = params.initial_epsilon;
    float delta = std::numeric_limits<float>::infinity();
    glm::vec2 uu = params.initial_control;
    std::array<glm::vec2, 2> dir = {glm::vec2{1.f, 0.f}, glm::vec2{0.f, 1.f}};
    while(iter < params.max_iter && delta > params.convergence_delta)
    {
        glm::vec2 g;
        switch(params.method)
        {
        case Method::FDSA: {
            float g0 = (0.5f / eps) * (combine(loss(image, palette, uu + eps * dir[0])) -
                                       combine(loss(image, palette, uu - eps * dir[0])));
            float g1 = (0.5f / eps) * (combine(loss(image, palette, uu + eps * dir[1])) -
                                       combine(loss(image, palette, uu - eps * dir[1])));
            g = {g0, g1};
            break;
        }
        case Method::SPSA: {
            glm::vec2 rvec{dis(gen), dis(gen)};
            rvec = glm::normalize(rvec);
            float g0 = (0.5f / (eps * rvec[0])) * (combine(loss(image, palette, uu + eps * rvec[0])) -
                                                   combine(loss(image, palette, uu - eps * rvec[0])));
            float g1 = (0.5f / (eps * rvec[1])) * (combine(loss(image, palette, uu + eps * rvec[1])) -
                                                   combine(loss(image, palette, uu - eps * rvec[1])));
            g = {g0, g1};
            break;
        }
        }

        auto old_uu = uu;
        uu -= ste * g;
        delta = glm::distance(uu, old_uu);

        KLOG("pencel", 1) << "Iteration: " << iter << std::endl;
        KLOGI << "epsilon: " << eps << " step: " << ste << " delta: " << delta << std::endl;
        KLOGI << "control: [" << uu[0] << ',' << uu[1] << ']' << std::endl;

        eps *= 0.97f;
        ste *= 0.97f;
        ++iter;
    }
    return uu;
}

glm::vec3 HSLOptimizer::loss(const Image& image, const std::vector<PencilInfo>& palette, const glm::vec2& control)
{
	// The fidelity loss is computed using the faster CMetric color difference

    constexpr float max_dist = 764.833f; // Maximal CMETRIC difference (black and white)
    auto argb_cmp = [](math::argb32_t a, math::argb32_t b) { return a.value < b.value; };

    float fidelity_loss = 0.f;
    std::set<math::argb32_t, decltype(argb_cmp)> colors_used;
    for(unsigned int row = 0; row < image.height; ++row)
    {
        for(unsigned int col = 0; col < image.width; ++col)
        {
            auto* pixel = BLOCK_OFFSET_RGB24(image.pixels.data(), image.width, col, row);
            auto value = math::pack_ARGB(pixel[0], pixel[1], pixel[2]);
            auto bm = best_match(value, palette, DeltaE::CMETRIC, control);
            fidelity_loss += bm.distance;
            if(bm.heavy)
                colors_used.insert(palette[bm.index].heavy_trace);
            else
                colors_used.insert(palette[bm.index].light_trace);
        }
    }

    // Low overall perceptive color difference w.r.t original image is penalized
    fidelity_loss /= (max_dist * float(image.height * image.width));
    // Low color diversity is penalized
    float diversity_loss = 1.f - float(colors_used.size()) / float(2 * palette.size());
    // Heavy HSL manipulation is penalized
    float transform_penalty = glm::distance(control, {1.f, 1.f});
    return {fidelity_loss, diversity_loss, transform_penalty};
}

float HSLOptimizer::combine(glm::vec3 loss_vector)
{
	// return (loss_vector[0] + loss_vector[1] + loss_vector[2]) / 3.f;
	return glm::length(loss_vector);
}
