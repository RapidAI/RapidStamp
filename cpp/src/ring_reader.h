#pragma once
// PARSeq whole-crop ring reader: lexicon rescoring + length routing.
// Mirrors python/src/rapidstamp/ring_reader.py. See route.json keys there.

#include "geometry.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rs {

struct NameLM {
    std::unordered_map<std::string, int> uni, bi, tri;
    long long total = 0;
    void build(const std::vector<std::string>& names);
    double logp(const std::string& left, const std::string& mid, const std::string& ch) const;
};

struct RingConfig {
    std::string main = "reader_main.onnx";
    std::string long_model;
    std::string charset = "charset.txt";
    std::string names;
    int long_threshold = 0;
    double alpha = 0.4;
    double tau = 4.0;
    int max_flips = 3;
    int image_size = 512;
    std::vector<std::string> classes;
};

struct RingOutput {
    std::string text;
    std::string main_text;
    bool routed = false;
    float score = 0.f;
};

std::vector<std::string> utf8_chars(const std::string& text);
bool parse_route_json(const std::string& text, RingConfig& config);
// PIL Image.resize(BILINEAR) on 8-bit RGB (Pillow Resample.c fixed point).
RsImage pil_resize_bilinear(const RsImage& src, int out_w, int out_h);

class RingReader {
public:
    // names_override: empty uses route.json "names". threshold < 0 uses route.json.
    RingReader(Ort::Env& env, const std::string& release_dir, int threads, const std::string& names_override = "",
               int threshold_override = -1);
    RingOutput read(const RsImage& crop);
    const RingConfig& config() const { return config_; }
    bool has_long() const { return static_cast<bool>(long_); }

    std::string decode(const float* logp, int len, int vocab, float* score) const;

private:
    std::vector<float> preprocess(const RsImage& crop) const;
    std::vector<float> run(Ort::Session& session, const std::vector<float>& input, int& len, int& vocab);
    std::string ids_text(const std::vector<int>& ids, int* count) const;

    RingConfig config_;
    std::vector<std::string> chars_;
    NameLM lm_;
    bool has_lm_ = false;
    std::unique_ptr<Ort::Session> main_;
    std::unique_ptr<Ort::Session> long_;
};

}  // namespace rs
