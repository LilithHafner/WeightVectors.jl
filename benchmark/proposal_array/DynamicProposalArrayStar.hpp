#pragma once
#include <cassert>
#include <functional>
#include <queue>
#include <random>
#include <vector>

namespace sampling {

class DynamicProposalArrayStar {
public:
    DynamicProposalArrayStar(const std::vector<double>& weights) :
        weights_(weights), R_(weights.size()), real_dist_(0, 1) {
        assert(weights.size() > 0);
        N_ = weights.size();
        W_ = std::accumulate(weights.begin(), weights.end(), 0.0);
        avg_ = W_ / N_;
        prev_avg_ = avg_;
        s_ = 0;
        cur_ = true;
        P1_.reserve(2 * N_);
        P2_.reserve(2 * N_);
        construct();
    }

    template <typename Generator>
    size_t sample(Generator&& gen) {
        auto& P_cur = cur_ ? P1_ : P2_;
        auto& P_nxt = cur_ ? P2_ : P1_;
        size_t buckets;
        if (s_ >= 0) buckets = R_.size() + P_cur.size() + 2 * P_nxt.size();
        else buckets = 2 * R_.size() + 2 * P_cur.size() + P_nxt.size();
        std::uniform_int_distribution<size_t> entry_dist(0, buckets - 1);
        do {
            size_t l = entry_dist(gen);
            size_t i = (s_ >= 0) ? l : l / 2;
            if (i < R_.size()) {
                double p_acc = R_[i];
                if (s_ > 0 && i < s_) {
                    p_acc *= 2;
                } else if (s_ < 0 && i < -s_) {
                    p_acc /= 2;
                }
                if (real_dist_(gen) < p_acc) {
                    return i;
                }
            } else {
                if (s_ >= 0) {
                    if (i < R_.size() + P_cur.size()) {
                        return P_cur[i - R_.size()].first;
                    } else {
                        return P_nxt[(i - R_.size() - P_cur.size()) / 2].first;
                    }
                } else {
                    if (i < R_.size() + P_cur.size()) {
                        return P_cur[i - R_.size()].first;
                    } else {
                        return P_nxt[2 * (i - R_.size() - P_cur.size())].first;
                    }
                }
            }
        } while (true);
    }

    void update(size_t i, double w) {
        double w_old = weights_[i];
        W_ += w - w_old;
        weights_[i] = w;

        double avg_power = (s_ > 0 && i < s_) ? avg_ * 2 : (s_ < 0 && i < -s_) ? avg_ / 2 : avg_;
        bool d = (s_ > 0 && i < s_) || (s_ < 0 && i < -s_);
        if (w > w_old) {
            size_t count = std::floor(w / avg_power);
            size_t old_count = L_[i].size();
            for (size_t c = old_count; c < count; ++c) {
                insert(i, !d);
            }
            R_[i] = (w / avg_power) - count;
        } else if (w < w_old) {
            size_t count = std::floor(w / avg_power);
            size_t old_count = L_[i].size();
            for (size_t c = old_count; c > count; --c) {
                erase(i, !d);
            }
            R_[i] = (w / avg_power) - count;
        }

        int64_t steps = 3 * N_ * std::log2((W_ / N_) / prev_avg_);
        if (W_ / N_ > prev_avg_) steps++;
        if (W_ / N_ < prev_avg_) steps--;
        prev_avg_ = W_ / N_;

        while (steps > 0 && s_ < N_) {
            bool d = s_ < 0;
            int64_t j = d ? -s_ - 1 : s_;
            double next_power = d ? avg_ : avg_ * 2;
            double weight = weights_[j];
            size_t count = std::floor(weight / next_power);
            size_t old_count = L_[j].size();
            for (size_t c = 0; c < old_count; ++c) {
                erase(j, !d);
                if (steps > 0) steps--;
            }
            for (size_t c = 0; c < count; ++c) {
                insert(j, d);
                if (steps > 0) steps--;
            }
            R_[j] = (weight / next_power) - count;
            s_++;
        }
        if (s_ >= N_ && W_ / N_ > 2 * avg_) {
            avg_ *= 2;
            s_ = 0;
            cur_ = !cur_;
        }
        while (steps < 0 && -s_ < N_) {
            bool d = s_ > 0;
            int64_t j = d ? s_ - 1 : -s_;
            double next_power = d ? avg_ : avg_ / 2;
            double weight = weights_[j];
            size_t count = std::floor(weight / next_power);
            size_t old_count = L_[j].size();
            for (size_t c = 0; c < old_count; ++c) {
                erase(j, !d);
                if (steps < 0) steps++;
            }
            for (size_t c = 0; c < count; ++c) {
                insert(j, d);
                if (steps < 0) steps++;
            }
            R_[j] = (weight / next_power) - count;
            s_--;
        }
        if (-s_ >= N_ && W_ / N_ < avg_ / 2) {
            avg_ /= 2;
            s_ = 0;
            cur_ = !cur_;
        }
    }

    size_t push(double w) {
        size_t i = weights_.size();
        weights_.push_back(0.0);
        R_.push_back(0.0);
        L_.push_back(std::vector<size_t>());
        N_++;
        update(i, w);
        return i;
    }

    void pop() {
        assert(weights_.size() > 0);
        size_t i = weights_.size() - 1;
        update(i, 0.0);
        weights_.pop_back();
        R_.pop_back();
        L_.pop_back();
        N_--;
    }

private:
    void construct() {
        L_ = std::vector<std::vector<size_t>>(N_);
        for (size_t i = 0; i < N_; ++i) {
            double weight = weights_[i];
            size_t count = std::floor(weight / avg_);
            for (size_t j = 0; j < count; ++j) {
                insert(i, true);
            }
            R_[i] = (weight / avg_) - count;
        }
    }

    void insert(size_t i, bool d) {
        if (!cur_) d = !d;
        auto& P_ = d ? P1_ : P2_;
        L_[i].push_back(P_.size());
        P_.emplace_back(i, L_[i].size() - 1);
    }

    void erase(size_t i, bool d) {
        assert(L_[i].size() > 0);
        if (!cur_) d = !d;
        auto& P_ = d ? P1_ : P2_;
        assert(P_.size() > 0);
        P_[L_[i].back()] = P_.back();
        L_[P_.back().first][P_.back().second] = L_[i].back();
        P_.pop_back();
        L_[i].pop_back();
    }

    std::vector<double> weights_;
    std::vector<double> R_;
    std::vector<std::pair<size_t, size_t>> P1_;
    std::vector<std::pair<size_t, size_t>> P2_;
    std::vector<std::vector<size_t>> L_;
    std::uniform_real_distribution<double> real_dist_;
    size_t N_;
    double W_;
    double avg_;
    double prev_avg_;
    int64_t s_;
    bool cur_;
};

}
