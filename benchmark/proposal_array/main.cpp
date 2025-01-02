#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "DynamicProposalArrayStar.hpp"

// Setup function for the sampler
sampling::DynamicProposalArrayStar setup_sampler(size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<double> weight_dist(1.0, 100.0);
    std::vector<double> weights(size);

    // Randomly initialize weights
    for (size_t i = 0; i < size; ++i) {
        weights[i] = weight_dist(rng);
    }

    return sampling::DynamicProposalArrayStar(weights);
}

// Fixed sampling benchmark
std::vector<size_t> benchmark_sample_fixed(sampling::DynamicProposalArrayStar& sampler, std::mt19937& rng, size_t n) {
    std::vector<size_t> samples;
    samples.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        samples.push_back(sampler.sample(rng));
    }

    return samples;
}

// Variable sampling benchmark
std::vector<size_t> benchmark_sample_variable(sampling::DynamicProposalArrayStar& sampler, std::mt19937& rng, size_t n) {
    std::vector<size_t> samples;
    samples.reserve(n);

    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    std::uniform_real_distribution<double> weight_dist(0.0, 1e6);

    for (size_t i = 0; i < n; ++i) {
        size_t index = index_dist(rng);
        sampler.update(index, 0.0);
        samples.push_back(sampler.sample(rng));
        double new_weight = weight_dist(rng);
        sampler.update(index, new_weight);
    }

    return samples;
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int exp = 3; exp <= 8; ++exp) {
        size_t size = static_cast<size_t>(std::pow(10, exp));

        // Setup sampler and measure time
        auto setup_start = std::chrono::high_resolution_clock::now();
        auto sampler = setup_sampler(size, rng);
        auto setup_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> setup_time = setup_end - setup_start;

        // Benchmark fixed sampling
        auto fixed_start = std::chrono::high_resolution_clock::now();
        auto fixed_samples = benchmark_sample_fixed(sampler, rng, size);
        auto fixed_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> fixed_time = fixed_end - fixed_start;

        // Benchmark variable sampling
        auto variable_start = std::chrono::high_resolution_clock::now();
        auto variable_samples = benchmark_sample_variable(sampler, rng, size);
        auto variable_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> variable_time = variable_end - variable_start;

        std::cout << "Size: " << size
                  << ", Setup time: " << setup_time.count() / size << " ns/element"
                  << ", Fixed sampling: " << fixed_time.count() / size << " ns/sample"
                  << ", Variable sampling: " << variable_time.count() / size << " ns/sample"
                  << std::endl;

        // Optional: Print sampled indices
        // std::cout << "Fixed samples: ";
        // for (auto idx : fixed_samples) {
        //     std::cout << idx << " ";
        // }
        // std::cout << "\nVariable samples: ";
        // for (auto idx : variable_samples) {
        //     std::cout << idx << " ";
        // }
        // std::cout << std::endl;
    }

    return 0;
}

