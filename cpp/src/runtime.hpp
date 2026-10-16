#pragma once

#include <string>
#include <vector>

namespace rsp {

struct Op {
    enum Kind { Conv, Relu, MaxPool, ToSeq, Gap, Linear };
    Kind kind;
    int args[9]{};
    std::vector<float> w;
    std::vector<float> b;
};

struct Model {
    std::string kind;
    int input_n = 1;
    int input_c = 0;
    int input_h = 0;
    int input_w = 0;
    std::vector<std::pair<std::string, std::string>> meta;
    std::vector<Op> ops;
    std::vector<std::string> classes;
    std::vector<std::string> words;

    std::string meta_str(const std::string& key, const std::string& fallback) const;
    float meta_float(const std::string& key, float fallback) const;
    int meta_int(const std::string& key, int fallback) const;
};

struct Tensor {
    int rank = 0;
    int d[4]{};
    std::vector<float> data;

    int size() const;
};

Model load_model(const std::string& manifest_path);
Tensor run_model(const Model& model, const float* nchw, int n, int c, int h, int w);

}  // namespace rsp
