// Code of the paper:
//
// @article{ZhangJW23,
//  author    = {Fangyuan Zhang and
//               Mengxu Jiang and
//               Sibo Wang},
//  title     = { Efficient Dynamic Weighted Set Sampling and Its Extension },
//  journal   = { Proc. {VLDB} Endow. },
//  volume    = { 17 },
//  number    = { 1 },
//  pages     = { 15-27 },
//  year      = { 2023 },
// }
//
// originally hosted at https://github.com/CUHK-DBGroup/WSS-WIRS

#pragma once

#include "utility.h"
#include <queue>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include "XoshiroCpp.hpp"
#include <unordered_map>
using namespace std;

class aliasMethod {
public:
    string method_name() {
        return "basic_alias";
    }
    aliasMethod(): gen(rd()) { constexpr std::uint64_t seed = 777; rng = XoshiroCpp::Xoroshiro128Plus(seed);
    }

    void reconstruct(int num, vector<Element>& all_ele) {
        alias.clear();
        prob_bias.clear();
        position_map.clear();

        elements.clear();

        for (int i = 0; i < num; i++) {
            elements.push_back(all_ele[i]);
            position_map[all_ele[i].key] = i;
        }
        ele_size = num;
        chrono::time_point<chrono::system_clock> start, end;
        chrono::duration<double> elapsed_seconds;
        //start = chrono::system_clock::now();
        init();
        //end = chrono::system_clock::now();
        //elapsed_seconds = end - start;
        //cout << "Alias init:" << elapsed_seconds.count() << "\n";
        reconstruct_flag = 0;
    }
    aliasMethod(int num, vector<Element> &all_ele) : gen(rd()){
        constexpr std::uint64_t seed = 777;
        rng = XoshiroCpp::Xoroshiro128Plus(seed);
        for (int i = 0; i < num; i++) {
            elements.push_back(all_ele[i]);
            position_map[all_ele[i].key] = i;
            tot_weight+=all_ele[i].weight;
        }
        ele_size = num;
        chrono::time_point<chrono::system_clock> start, end;
        chrono::duration<double> elapsed_seconds;
        //start = chrono::system_clock::now();
        init();
        //end = chrono::system_clock::now();
        //elapsed_seconds = end - start;
        //cout << "Alias init:" << elapsed_seconds.count() << "\n";
        reconstruct_flag = 0;
    }
    //randomly generate
    int random_sample_value() {
        if (reconstruct_flag == 1) {
            init();
            reconstruct_flag = 0;
        }
        pos_rand = uniform_int_distribution<int>(0, ele_size - 1);
        dr = uniform_real_distribution<double>(0, 1.0);

        int sel = pos_rand(rng);
        double sel2 = dr(rng);
        //cerr<<sel<<" " << sel2 << " " << sel << "\n";
        if (sel2 < prob_bias[sel]) return elements[sel].value;
        else return elements[alias[sel]].value;
    }


    void delete_element(int del_key) {
        reconstruct_flag = 1;

        int del_pos = position_map[del_key];
        int tail_key = elements[ele_size - 1].key;
        elements[del_pos] = elements[ele_size - 1];
        position_map[tail_key] = del_pos;
        position_map.erase(del_key);
        assert(!find_key(del_key));
        elements.pop_back();
        ele_size--;
    }

    bool find_key(int tmp_key) {
        return (position_map.find(tmp_key) != position_map.end());
    }

    void insert_element(Element& ins_ele) {
        reconstruct_flag = 1;
        if (find_key(ins_ele.key)) {
            cout <<ins_ele.key<<" " << "insert an existed element!!!!!!!!!\n";
        }
        position_map[ins_ele.key] = ele_size;
        elements.push_back(ins_ele);
        ele_size++;
    }
//private:
public:
    int ele_size, reconstruct_flag;
    random_device rd;
    XoshiroCpp::Xoroshiro128Plus rng;

    double tot_weight = 0;
    mt19937 gen;
    uniform_int_distribution<int> pos_rand;
    uniform_real_distribution<double> dr;
    //int* alias;
    vector<int> alias;
    vector<double> prob_bias;
    vector<Element> elements;
    unordered_map<int, int> position_map;
    //double* prob_bias;

    void init() {
        alias.resize(ele_size);
        prob_bias.resize(ele_size);
        queue<int> small;
        queue<int> large;
        vector<double> tmp_p;
        tmp_p.resize(ele_size);
        int i;

        double sum = 0;

        for (i = 0; i < ele_size; i++) sum += elements[i].weight;
        //cerr << sum << "\n";
        for (i = 0; i < ele_size; i++)
        {
            tmp_p[i] = elements[i].weight * ele_size / sum;
            if (tmp_p[i] < 1)small.push(i);
            else large.push(i);
        }
        while ((!small.empty()) && (!large.empty()))
        {
            int l = small.front();
            small.pop();
            int g = large.front();
            large.pop();
            prob_bias[l] = tmp_p[l];
            alias[l] = g;
            tmp_p[g] = tmp_p[g] + tmp_p[l] - 1;
            if (tmp_p[g] < 1)small.push(g);
            else large.push(g);
        }
        while (!large.empty())
        {
            int g = large.front();
            large.pop();
            prob_bias[g] = 1;
            alias[g] = -1;
        }
        while (!small.empty())
        {
            int l = small.front();
            small.pop();
            prob_bias[l] = 1;
            alias[l] = -1;
        }

    }
};
