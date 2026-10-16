#include "runtime.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rsp {
namespace {

std::vector<float> read_f32(const std::string& path, int count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open weights: " + path);
    }
    std::vector<float> data(static_cast<size_t>(count));
    if (count > 0) {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (in.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
            throw std::runtime_error("short weight file: " + path);
        }
    }
    return data;
}

Tensor conv2d(const Tensor& x, const Op& op) {
    const int in_c = op.args[0];
    const int out_c = op.args[1];
    const int kh = op.args[2];
    const int kw = op.args[3];
    const int sh = op.args[4];
    const int sw = op.args[5];
    const int ph = op.args[6];
    const int pw = op.args[7];
    const int groups = op.args[8];
    const int n = x.d[0];
    const int c = x.d[1];
    const int h = x.d[2];
    const int w = x.d[3];
    if (c != in_c) {
        throw std::runtime_error("conv channel mismatch");
    }
    const int ic_g = in_c / groups;
    const int oc_g = out_c / groups;
    const int hp = h + 2 * ph;
    const int wp = w + 2 * pw;
    std::vector<float> padded(static_cast<size_t>(n * c * hp * wp), 0.f);
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            for (int y = 0; y < h; ++y) {
                const float* src = &x.data[((ni * c + ci) * h + y) * w];
                float* dst = &padded[((ni * c + ci) * hp + (y + ph)) * wp + pw];
                std::copy(src, src + w, dst);
            }
        }
    }
    const int oh = (hp - kh) / sh + 1;
    const int ow = (wp - kw) / sw + 1;
    Tensor y;
    y.rank = 4;
    y.d[0] = n;
    y.d[1] = out_c;
    y.d[2] = oh;
    y.d[3] = ow;
    y.data.assign(static_cast<size_t>(n * out_c * oh * ow), 0.f);
    const float* weight = op.w.data();
    const float* bias = op.b.data();
    for (int ni = 0; ni < n; ++ni) {
        for (int g = 0; g < groups; ++g) {
            for (int oc = 0; oc < oc_g; ++oc) {
                const int o = g * oc_g + oc;
                for (int oy = 0; oy < oh; ++oy) {
                    for (int ox = 0; ox < ow; ++ox) {
                        float sum = bias[o];
                        for (int ic = 0; ic < ic_g; ++ic) {
                            const int ci = g * ic_g + ic;
                            for (int ky = 0; ky < kh; ++ky) {
                                const int iy = oy * sh + ky;
                                for (int kx = 0; kx < kw; ++kx) {
                                    const int ix = ox * sw + kx;
                                    const float xv = padded[((ni * c + ci) * hp + iy) * wp + ix];
                                    const int wi = ((o * ic_g + ic) * kh + ky) * kw + kx;
                                    sum += xv * weight[wi];
                                }
                            }
                        }
                        y.data[((ni * out_c + o) * oh + oy) * ow + ox] = sum;
                    }
                }
            }
        }
    }
    return y;
}

Tensor maxpool2d(const Tensor& x, const Op& op) {
    const int kh = op.args[0];
    const int kw = op.args[1];
    const int sh = op.args[2];
    const int sw = op.args[3];
    const int ph = op.args[4];
    const int pw = op.args[5];
    const int n = x.d[0];
    const int c = x.d[1];
    const int h = x.d[2];
    const int w = x.d[3];
    const int hp = h + 2 * ph;
    const int wp = w + 2 * pw;
    const float kPad = -INFINITY;
    std::vector<float> padded(static_cast<size_t>(n * c * hp * wp), kPad);
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            for (int y = 0; y < h; ++y) {
                const float* src = &x.data[((ni * c + ci) * h + y) * w];
                float* dst = &padded[((ni * c + ci) * hp + (y + ph)) * wp + pw];
                std::copy(src, src + w, dst);
            }
        }
    }
    const int oh = (hp - kh) / sh + 1;
    const int ow = (wp - kw) / sw + 1;
    Tensor y;
    y.rank = 4;
    y.d[0] = n;
    y.d[1] = c;
    y.d[2] = oh;
    y.d[3] = ow;
    y.data.assign(static_cast<size_t>(n * c * oh * ow), kPad);
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float m = kPad;
                    for (int ky = 0; ky < kh; ++ky) {
                        const int iy = oy * sh + ky;
                        for (int kx = 0; kx < kw; ++kx) {
                            const int ix = ox * sw + kx;
                            m = std::max(m, padded[((ni * c + ci) * hp + iy) * wp + ix]);
                        }
                    }
                    y.data[((ni * c + ci) * oh + oy) * ow + ox] = m;
                }
            }
        }
    }
    return y;
}

Tensor gap(const Tensor& x) {
    const int n = x.d[0];
    const int c = x.d[1];
    const int h = x.d[2];
    const int w = x.d[3];
    Tensor y;
    y.rank = 2;
    y.d[0] = n;
    y.d[1] = c;
    y.data.assign(static_cast<size_t>(n * c), 0.f);
    const float inv = 1.f / static_cast<float>(h * w);
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            float sum = 0.f;
            for (int hi = 0; hi < h; ++hi) {
                for (int wi = 0; wi < w; ++wi) {
                    sum += x.data[(((ni * c + ci) * h + hi) * w) + wi];
                }
            }
            y.data[static_cast<size_t>(ni * c + ci)] = sum * inv;
        }
    }
    return y;
}

Tensor to_seq(const Tensor& x) {
    const int n = x.d[0];
    const int c = x.d[1];
    const int h = x.d[2];
    const int w = x.d[3];
    Tensor y;
    y.rank = 3;
    if (h == 1) {
        y.d[0] = n;
        y.d[1] = w;
        y.d[2] = c;
        y.data.resize(static_cast<size_t>(n * w * c));
        for (int ni = 0; ni < n; ++ni) {
            for (int wi = 0; wi < w; ++wi) {
                for (int ci = 0; ci < c; ++ci) {
                    y.data[(ni * w + wi) * c + ci] = x.data[(ni * c + ci) * w + wi];
                }
            }
        }
        return y;
    }
    if (w == 1) {
        y.d[0] = n;
        y.d[1] = h;
        y.d[2] = c;
        y.data.resize(static_cast<size_t>(n * h * c));
        for (int ni = 0; ni < n; ++ni) {
            for (int hi = 0; hi < h; ++hi) {
                for (int ci = 0; ci < c; ++ci) {
                    y.data[(ni * h + hi) * c + ci] = x.data[((ni * c + ci) * h + hi)];
                }
            }
        }
        return y;
    }
    throw std::runtime_error("to_seq expected a singleton spatial axis");
}

Tensor linear(const Tensor& x, const Op& op) {
    const int in_f = op.args[0];
    const int out_f = op.args[1];
    int inner = x.d[x.rank - 1];
    if (inner != in_f) {
        throw std::runtime_error("linear input mismatch");
    }
    int outer = 1;
    for (int i = 0; i < x.rank - 1; ++i) {
        outer *= x.d[i];
    }
    Tensor y;
    y.rank = x.rank;
    for (int i = 0; i < x.rank - 1; ++i) {
        y.d[i] = x.d[i];
    }
    y.d[x.rank - 1] = out_f;
    y.data.resize(static_cast<size_t>(outer * out_f));
    for (int o = 0; o < outer; ++o) {
        const float* src = &x.data[o * in_f];
        for (int oc = 0; oc < out_f; ++oc) {
            float sum = op.b[static_cast<size_t>(oc)];
            const float* w = &op.w[static_cast<size_t>(oc * in_f)];
            for (int i = 0; i < in_f; ++i) {
                sum += src[i] * w[i];
            }
            y.data[o * out_f + oc] = sum;
        }
    }
    return y;
}

}  // namespace

int Tensor::size() const {
    if (rank <= 0) {
        return 0;
    }
    int n = 1;
    for (int i = 0; i < rank; ++i) {
        n *= d[i];
    }
    return n;
}

std::string Model::meta_str(const std::string& key, const std::string& fallback) const {
    for (const auto& item : meta) {
        if (item.first == key) {
            return item.second;
        }
    }
    return fallback;
}

float Model::meta_float(const std::string& key, float fallback) const {
    const std::string value = meta_str(key, "");
    if (value.empty()) {
        return fallback;
    }
    return std::stof(value);
}

int Model::meta_int(const std::string& key, int fallback) const {
    const std::string value = meta_str(key, "");
    if (value.empty()) {
        return fallback;
    }
    return std::stoi(value);
}

Model load_model(const std::string& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in) {
        throw std::runtime_error("cannot open manifest: " + manifest_path);
    }
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty manifest");
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != "rapidstamp 1") {
        throw std::runtime_error("not a rapidstamp v1 manifest");
    }
    Model model;
    int weight_count = 0;
    std::vector<Op> specs;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "kind") {
            ss >> model.kind;
        } else if (tag == "input") {
            ss >> model.input_n >> model.input_c >> model.input_h >> model.input_w;
        } else if (tag == "meta") {
            std::string key;
            std::string value;
            ss >> key >> value;
            model.meta.emplace_back(key, value);
        } else if (tag == "weight_count") {
            ss >> weight_count;
        } else if (tag == "classes") {
            continue;
        } else if (tag == "class") {
            std::string ch;
            ss >> ch;
            model.classes.push_back(ch);
        } else if (tag == "words") {
            continue;
        } else if (tag == "word") {
            std::string word;
            ss >> word;
            model.words.push_back(word);
        } else if (tag == "op") {
            std::string kind;
            ss >> kind;
            Op op;
            if (kind == "conv") {
                op.kind = Op::Conv;
                for (int i = 0; i < 9; ++i) {
                    ss >> op.args[i];
                }
            } else if (kind == "relu") {
                op.kind = Op::Relu;
            } else if (kind == "maxpool") {
                op.kind = Op::MaxPool;
                for (int i = 0; i < 6; ++i) {
                    ss >> op.args[i];
                }
            } else if (kind == "to_seq") {
                op.kind = Op::ToSeq;
            } else if (kind == "gap") {
                op.kind = Op::Gap;
            } else if (kind == "linear") {
                op.kind = Op::Linear;
                ss >> op.args[0] >> op.args[1];
            } else {
                throw std::runtime_error("unknown op " + kind);
            }
            specs.push_back(op);
        } else {
            throw std::runtime_error("bad manifest line: " + line);
        }
    }
    std::string bin_path = manifest_path;
    const auto dot = bin_path.rfind('.');
    if (dot != std::string::npos) {
        bin_path = bin_path.substr(0, dot);
    }
    bin_path += ".bin";
    std::vector<float> weights = read_f32(bin_path, weight_count);
    size_t cursor = 0;
    auto take = [&](int n) {
        if (cursor + static_cast<size_t>(n) > weights.size()) {
            throw std::runtime_error("unexpected end of weights");
        }
        std::vector<float> slab(weights.begin() + static_cast<long>(cursor), weights.begin() + static_cast<long>(cursor + n));
        cursor += static_cast<size_t>(n);
        return slab;
    };
    for (auto op : specs) {
        if (op.kind == Op::Conv) {
            const int in_c = op.args[0];
            const int out_c = op.args[1];
            const int kh = op.args[2];
            const int kw = op.args[3];
            const int groups = op.args[8];
            const int ic_g = in_c / groups;
            const int wn = out_c * ic_g * kh * kw;
            op.w = take(wn);
            op.b = take(out_c);
        } else if (op.kind == Op::Linear) {
            op.w = take(op.args[0] * op.args[1]);
            op.b = take(op.args[1]);
        }
        model.ops.push_back(std::move(op));
    }
    if (cursor != weights.size()) {
        throw std::runtime_error("unused weights");
    }
    return model;
}

Tensor run_model(const Model& model, const float* nchw, int n, int c, int h, int w) {
    if (c != model.input_c || h != model.input_h || w != model.input_w) {
        throw std::runtime_error("input shape does not match the model");
    }
    Tensor cur;
    cur.rank = 4;
    cur.d[0] = n;
    cur.d[1] = c;
    cur.d[2] = h;
    cur.d[3] = w;
    cur.data.assign(nchw, nchw + static_cast<size_t>(n * c * h * w));
    for (const Op& op : model.ops) {
        if (op.kind == Op::Conv) {
            cur = conv2d(cur, op);
        } else if (op.kind == Op::Relu) {
            for (float& v : cur.data) {
                if (v < 0.f) {
                    v = 0.f;
                }
            }
        } else if (op.kind == Op::MaxPool) {
            cur = maxpool2d(cur, op);
        } else if (op.kind == Op::ToSeq) {
            cur = to_seq(cur);
        } else if (op.kind == Op::Gap) {
            cur = gap(cur);
        } else if (op.kind == Op::Linear) {
            cur = linear(cur, op);
        }
    }
    return cur;
}

}  // namespace rsp
