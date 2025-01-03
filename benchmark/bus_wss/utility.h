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

struct Element {
	Element(const int& _key, const int& _value, const float& _weight) {
		key = _key;
		value = _value;
		weight = _weight;
	}
    Element(){}
	~Element() {}
	friend bool operator < (const Element& a, const Element& b) {
		if (a.weight > b.weight) {
			return true;
		}
		else if (a.weight < b.weight) {
			return false;
		}
		else {
			if (a.key > b.key)
				return true;
			else
				return false;
		}
	}
	int key; // searched by key
	float weight;
	int value; 
};

struct Opt {
	Opt(int _opt_type, int _key, int _value, float _weight) {
		opt_type = _opt_type;
		key = _key;
		value = _value;
		weight = _weight;
	}
	~Opt() {}
	int opt_type; // searched by key
	//float weight;
	int key;
	int value;
	float weight;
	//Element ins_ele;
};

bool cmp_element_weight(Element& a, Element& b) {
	if (a.weight > b.weight) {
		return true;
	}
	else if(a.weight < b.weight) {
		return false;
	}
	else {
		if (a.key > b.key)
			return true;
		else
			return false;
	}
}

struct cmp_element_key_struct
{
    bool operator()(const Element& i, const Element& j) {
        return i.key < j.key;
    }
};

bool cmp_element_key(Element& a, Element& b) {
        return a.key < b.key;
}