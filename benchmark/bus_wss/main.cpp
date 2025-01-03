
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "XoshiroCpp.hpp"
#include "QuickBucket.hpp"

using namespace std;

// Function to set up the BucketMethod with random weights
BucketMethod setup(XoshiroCpp::Xoroshiro128Plus& rng, int n) {
    vector<Element> elements;
    elements.reserve(n);

    uniform_real_distribution<double> weight_dist(0.0, pow(10.0,30));
    for (int i = 0; i < n; ++i) {
        Element e(i, i, weight_dist(rng));
        elements.emplace_back(e);
    }

    return BucketMethod(n, elements);
}

vector<int> sample_fixed(XoshiroCpp::Xoroshiro128Plus& rng, BucketMethod& bucket, int n) {
    vector<int> samples;
    samples.reserve(n);

    for (int i = 0; i < n; ++i) {
        samples.push_back(bucket.random_sample_value());
    }

    return samples;
}

vector<int> sample_variable(XoshiroCpp::Xoroshiro128Plus& rng, BucketMethod& bucket, int n) {
    vector<int> samples;
    samples.reserve(n);

    uniform_real_distribution<double> weight_dist(0.0, pow(10.0,30));
    uniform_int_distribution<int> int_dist(0, n - 1);

    for (int i = 0; i < n; ++i) {
        int j = int_dist(rng);
        bucket.delete_element(j);
        samples.push_back(bucket.random_sample_value());
        Element e(j, j, weight_dist(rng));
        bucket.insert_element(e);
    }
    
    return samples;
}

// Benchmarking the setup and sampling functions
void benchmark() {
    constexpr uint64_t seed = 42;
    XoshiroCpp::Xoroshiro128Plus rng(seed);

    for (int s_exp = 3; s_exp <= 8; ++s_exp) {
        int s = static_cast<int>(pow(10, s_exp));

        // Benchmark fixed sampling
        BucketMethod bucket_fixed = setup(rng, s);
        auto start_fixed = chrono::high_resolution_clock::now();
        sample_fixed(rng, bucket_fixed, s);
        auto end_fixed = chrono::high_resolution_clock::now();
        double time_fixed = chrono::duration_cast<chrono::nanoseconds>(end_fixed - start_fixed).count() / static_cast<double>(s);

        // Benchmark variable sampling
        BucketMethod bucket_variable = setup(rng, s);
        auto start_variable = chrono::high_resolution_clock::now();
        sample_variable(rng, bucket_variable, s);
        auto end_variable = chrono::high_resolution_clock::now();
        double time_variable = chrono::duration_cast<chrono::nanoseconds>(end_variable - start_variable).count() / static_cast<double>(s);

        cout << time_fixed << " " << time_variable << endl;
    }
}

int main() {
    benchmark();
    return 0;
}

