#include "ring_reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rs {
namespace {

const int kPad = 0, kBos = 1, kEos = 2;

const char* const kSuffixes[] = {
    "有限责任公司", "股份有限公司", "集团有限公司", "有限公司", "发票监制章", "人民政府", "监督管理局",
    "市场监督管理局", "委员会", "办事处", "办公室", "管理局", "公安局", "检察院", "法院", "卫生院", "事务所",
    "合作社", "研究院", "研究所", "服务中心", "交易中心", "分公司", "支行", "分行", "大学", "学院", "学校",
    "医院", "银行", "中心", "集团", "政府", "幼儿园", "经营部", "管理处", "信用社", "公司", "局", "厅", "院",
    "所", "站", "队", "部", "社", "厂", "店", "馆", "会", "署", "处", "自治区", "自治州", "街道", "省", "市",
    "区", "县", "镇", "乡", "村"};

std::string slurp(const std::string& path) {
#ifdef _WIN32
    int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<size_t>(std::max(size, 1)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), size);
    std::ifstream in(wide.c_str(), std::ios::binary);
#else
    std::ifstream in(path, std::ios::binary);
#endif
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

std::vector<std::string> split_lines(const std::string& text, bool strip) {
    std::vector<std::string> out;
    std::string line;
    std::istringstream in(text);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (strip) {
            size_t a = line.find_first_not_of(" \t");
            size_t b = line.find_last_not_of(" \t");
            line = a == std::string::npos ? "" : line.substr(a, b - a + 1);
        }
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

std::string key2(const std::string& a, const std::string& b) { return a + '\x1f' + b; }
std::string key3(const std::string& a, const std::string& b, const std::string& c) {
    return a + '\x1f' + b + '\x1f' + c;
}

int count_of(const std::unordered_map<std::string, int>& m, const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
}

std::unique_ptr<Ort::Session> open(Ort::Env& env, const std::string& path, const Ort::SessionOptions& options) {
#ifdef _WIN32
    int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<size_t>(std::max(size, 1)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), size);
    return std::make_unique<Ort::Session>(env, wide.c_str(), options);
#else
    return std::make_unique<Ort::Session>(env, path.c_str(), options);
#endif
}

// --- tiny flat-JSON reader for route.json ---
void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ',')) ++i;
}
bool read_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    out.clear();
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) ++i;
        out.push_back(s[i++]);
    }
    ++i;
    return true;
}

// Pillow Resample.c, bilinear, 8 bpc.
const int kPrecision = 32 - 8 - 2;
struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;  // xmin, xsize pairs
    std::vector<int> k;       // fixed point
};
Coeffs precompute(int in_size, int out_size) {
    double scale = static_cast<double>(in_size) / out_size;
    double filterscale = std::max(scale, 1.0);
    double support = 1.0 * filterscale;
    int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    Coeffs c;
    c.ksize = ksize;
    c.bounds.resize(static_cast<size_t>(out_size) * 2);
    c.k.assign(static_cast<size_t>(out_size) * ksize, 0);
    std::vector<double> kk(ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        double center = (xx + 0.5) * scale;
        double ww = 0.0, ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; ++x) {
            double t = (x + xmin - center + 0.5) * ss;
            if (t < 0) t = -t;
            double w = t < 1.0 ? 1.0 - t : 0.0;
            kk[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x) {
            double v = ww != 0.0 ? kk[x] / ww : kk[x];
            c.k[static_cast<size_t>(xx) * ksize + x] =
                static_cast<int>(v < 0 ? std::trunc(-0.5 + v * (1 << kPrecision)) : std::trunc(0.5 + v * (1 << kPrecision)));
        }
        c.bounds[xx * 2] = xmin;
        c.bounds[xx * 2 + 1] = xmax;
    }
    return c;
}
inline uint8_t clip8(long long v) {
    v >>= kPrecision;
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

}  // namespace

std::vector<std::string> utf8_chars(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        out.push_back(text.substr(i, n));
        i += n;
    }
    return out;
}

RsImage pil_resize_bilinear(const RsImage& src, int out_w, int out_h) {
    RsImage tmp = src;
    if (out_w != src.width) {
        Coeffs c = precompute(src.width, out_w);
        RsImage h;
        h.width = out_w;
        h.height = src.height;
        h.channels = 3;
        h.data.resize(static_cast<size_t>(out_w) * src.height * 3);
        for (int y = 0; y < src.height; ++y) {
            for (int xx = 0; xx < out_w; ++xx) {
                int xmin = c.bounds[xx * 2], xs = c.bounds[xx * 2 + 1];
                const int* k = &c.k[static_cast<size_t>(xx) * c.ksize];
                for (int ch = 0; ch < 3; ++ch) {
                    long long ss = 1LL << (kPrecision - 1);
                    for (int x = 0; x < xs; ++x) ss += static_cast<long long>(src.pixel(x + xmin, y)[ch]) * k[x];
                    h.pixel(xx, y)[ch] = clip8(ss);
                }
            }
        }
        tmp = std::move(h);
    }
    if (out_h != tmp.height) {
        Coeffs c = precompute(tmp.height, out_h);
        RsImage v;
        v.width = tmp.width;
        v.height = out_h;
        v.channels = 3;
        v.data.resize(static_cast<size_t>(tmp.width) * out_h * 3);
        for (int yy = 0; yy < out_h; ++yy) {
            int ymin = c.bounds[yy * 2], ys = c.bounds[yy * 2 + 1];
            const int* k = &c.k[static_cast<size_t>(yy) * c.ksize];
            for (int x = 0; x < tmp.width; ++x) {
                for (int ch = 0; ch < 3; ++ch) {
                    long long ss = 1LL << (kPrecision - 1);
                    for (int y = 0; y < ys; ++y) ss += static_cast<long long>(tmp.pixel(x, y + ymin)[ch]) * k[y];
                    v.pixel(x, yy)[ch] = clip8(ss);
                }
            }
        }
        tmp = std::move(v);
    }
    return tmp;
}

bool parse_route_json(const std::string& s, RingConfig& config) {
    size_t i = s.find('{');
    if (i == std::string::npos) return false;
    ++i;
    while (true) {
        skip_ws(s, i);
        if (i >= s.size() || s[i] == '}') break;
        std::string key;
        if (!read_string(s, i, key)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') return false;
        ++i;
        skip_ws(s, i);
        if (i < s.size() && s[i] == '"') {
            std::string value;
            read_string(s, i, value);
            if (key == "main") config.main = value;
            else if (key == "long") config.long_model = value;
            else if (key == "charset") config.charset = value;
            else if (key == "names") config.names = value;
        } else if (i < s.size() && s[i] == '[') {
            ++i;
            std::vector<std::string> items;
            while (true) {
                skip_ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ']') { ++i; break; }
                std::string value;
                if (!read_string(s, i, value)) return false;
                items.push_back(value);
            }
            if (key == "classes") config.classes = items;
        } else {
            size_t j = i;
            while (j < s.size() && s[j] != ',' && s[j] != '}' && s[j] != '\n') ++j;
            std::string raw = s.substr(i, j - i);
            i = j;
            if (raw.find("null") != std::string::npos) continue;
            double value = std::stod(raw);
            if (key == "long_threshold") config.long_threshold = static_cast<int>(value);
            else if (key == "alpha") config.alpha = value;
            else if (key == "tau") config.tau = value;
            else if (key == "max_flips") config.max_flips = static_cast<int>(value);
            else if (key == "image_size") config.image_size = static_cast<int>(value);
        }
    }
    return true;
}

void NameLM::build(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        auto chars = utf8_chars(name);
        if (chars.empty()) continue;
        total += static_cast<long long>(chars.size());
        bi[key2("^", "^")] += 1;
        std::string a = "^", b = "^";
        for (const auto& c : chars) {
            uni[c] += 1;
            bi[key2(b, c)] += 1;
            tri[key3(a, b, c)] += 1;
            a = b;
            b = c;
        }
    }
}

double NameLM::logp(const std::string& left, const std::string& mid, const std::string& ch) const {
    double n_ab = left.empty() ? 0 : count_of(bi, key2(left, mid));
    double n_abc = count_of(tri, key3(left, mid, ch));
    double n_b = mid != "^" ? count_of(uni, mid) : static_cast<double>(total);
    double n_bc = count_of(bi, key2(mid, ch));
    double p_tri = (n_ab != 0 || n_abc != 0) ? (n_abc + 0.05) / (n_ab + 0.05 * 40.0) : 0.0;
    double p_bi = (n_b != 0 || n_bc != 0) ? (n_bc + 0.05) / (n_b + 0.05 * 40.0) : 0.0;
    double p_uni = (count_of(uni, ch) + 0.05) / (static_cast<double>(total) + 0.05 * 40.0);
    double mixed;
    if (p_tri == 0.0 && p_bi == 0.0) mixed = p_uni;
    else if (p_tri == 0.0) mixed = 0.75 * p_bi + 0.25 * p_uni;
    else mixed = 0.65 * p_tri + 0.25 * p_bi + 0.10 * p_uni;
    return std::log(std::max(mixed, 1e-8));
}

RingReader::RingReader(Ort::Env& env, const std::string& dir, int threads, const std::string& names_override,
                       int threshold_override) {
    if (!parse_route_json(slurp(dir + "/route.json"), config_)) throw std::runtime_error("bad route.json");
    if (threshold_override >= 0) config_.long_threshold = threshold_override;
    // charset: keep a lone space line (it is a real character).
    {
        std::string text = slurp(dir + "/" + config_.charset);
        std::string line;
        std::istringstream in(text);
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) chars_.push_back(line);
        }
    }
    std::string names = names_override.empty() ? (config_.names.empty() ? "" : dir + "/" + config_.names) : names_override;
    if (!names.empty()) {
        lm_.build(split_lines(slurp(names), true));
        has_lm_ = true;
    }
    Ort::SessionOptions options;
    if (threads > 0) options.SetIntraOpNumThreads(threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    main_ = open(env, dir + "/" + config_.main, options);
    if (!config_.long_model.empty() && config_.long_threshold > 0) long_ = open(env, dir + "/" + config_.long_model, options);
}

std::vector<float> RingReader::preprocess(const RsImage& crop) const {
    int side = std::max({crop.width, crop.height, 1});
    RsImage canvas;
    canvas.width = canvas.height = side;
    canvas.channels = 3;
    canvas.data.assign(static_cast<size_t>(side) * side * 3, 255);
    int ox = (side - crop.width) / 2, oy = (side - crop.height) / 2;
    for (int y = 0; y < crop.height; ++y)
        std::copy(crop.pixel(0, y), crop.pixel(0, y) + crop.width * 3, canvas.pixel(ox, oy + y));
    int s = config_.image_size;
    RsImage img = pil_resize_bilinear(canvas, s, s);
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(static_cast<size_t>(3) * s * s);
    size_t plane = static_cast<size_t>(s) * s;
    for (int y = 0; y < s; ++y)
        for (int x = 0; x < s; ++x) {
            const uint8_t* p = img.pixel(x, y);
            for (int c = 0; c < 3; ++c)
                out[c * plane + static_cast<size_t>(y) * s + x] = (p[c] / 255.0f - mean[c]) / stdv[c];
        }
    return out;
}

std::vector<float> RingReader::run(Ort::Session& session, const std::vector<float>& input, int& len, int& vocab) {
    int s = config_.image_size;
    std::array<int64_t, 4> shape{1, 3, s, s};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(),
                                                        shape.data(), shape.size());
    const char* in_names[] = {"image"};
    const char* out_names[] = {"logprobs"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    len = static_cast<int>(dims[1]);
    vocab = static_cast<int>(dims[2]);
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + static_cast<size_t>(len) * vocab);
}

std::string RingReader::ids_text(const std::vector<int>& ids, int* count) const {
    std::string text;
    int n = 0;
    for (int id : ids) {
        if (id == kEos || id == kPad) break;
        if (id == kBos) continue;
        int j = id - 3;
        if (j >= 0 && j < static_cast<int>(chars_.size())) {
            text += chars_[j];
            n += static_cast<int>(utf8_chars(chars_[j]).size());
        }
    }
    if (count) *count = n;
    return text;
}

std::string RingReader::decode(const float* logp, int len, int vocab, float* score) const {
    std::vector<int> ids, runners;
    std::vector<double> gaps;
    bool ended = false;
    for (int t = 0; t < len; ++t) {
        const float* row = logp + static_cast<size_t>(t) * vocab;
        int top = 0;
        for (int v = 1; v < vocab; ++v) if (row[v] > row[top]) top = v;
        int second = top == 0 ? 1 : 0;
        for (int v = 0; v < vocab; ++v) if (v != top && row[v] > row[second]) second = v;
        // tie on equal values: lowest index wins (stable descending sort)
        for (int v = 0; v < vocab; ++v) if (v != top && row[v] == row[second]) { second = v; break; }
        if (top == kPad || top == kEos) { ended = true; break; }
        ids.push_back(top);
        runners.push_back(second);
        gaps.push_back(static_cast<double>(row[top]) - static_cast<double>(row[second]));
    }
    (void)ended;
    auto conf = [&](const std::vector<int>& chosen) {
        double sum = 0;
        for (size_t t = 0; t < chosen.size(); ++t) sum += logp[t * vocab + chosen[t]];
        return static_cast<float>(std::exp(sum / std::max<size_t>(chosen.size(), 1)));
    };
    int glen = 0;
    std::string greedy = ids_text(ids, &glen);
    if (config_.alpha == 0.0 || !has_lm_ || greedy.empty()) {
        if (score) *score = conf(ids);
        return greedy;
    }
    auto gchars = utf8_chars(greedy);
    size_t suffix_len = 0;
    for (const char* sfx : kSuffixes) {  // longest matching suffix
        auto sc = utf8_chars(sfx);
        if (sc.size() < gchars.size() && sc.size() > suffix_len &&
            std::equal(sc.begin(), sc.end(), gchars.end() - static_cast<long>(sc.size())))
            suffix_len = sc.size();
    }
    int stem_end = static_cast<int>(gchars.size() - suffix_len);
    std::vector<int> uncertain;
    for (int i = 0; i < stem_end; ++i) if (gaps[i] < config_.tau) uncertain.push_back(i);
    std::stable_sort(uncertain.begin(), uncertain.end(), [&](int a, int b) { return gaps[a] < gaps[b]; });
    if (static_cast<int>(uncertain.size()) > config_.max_flips) uncertain.resize(config_.max_flips);
    if (uncertain.empty()) {
        if (score) *score = conf(ids);
        return greedy;
    }
    auto score_ids = [&](const std::vector<int>& cand) {
        int n = 0;
        std::string text = ids_text(cand, &n);
        if (n != static_cast<int>(cand.size())) return -1e9;
        auto cs = utf8_chars(text);
        double model = 0, language = 0;
        std::string left = "^", mid = "^";
        for (size_t t = 0; t < cand.size(); ++t) {
            model += logp[t * vocab + cand[t]];
            language += lm_.logp(left, mid, cs[t]);
            left = mid;
            mid = cs[t];
        }
        double count = std::max<size_t>(cand.size(), 1);
        return model / count + config_.alpha * (language / count);
    };
    std::vector<int> best_ids = ids;
    double best = score_ids(ids);
    for (int mask = 0; mask < (1 << uncertain.size()); ++mask) {
        std::vector<int> cand = ids;
        for (size_t bit = 0; bit < uncertain.size(); ++bit)
            if (mask & (1 << bit)) cand[uncertain[bit]] = runners[uncertain[bit]];
        double value = score_ids(cand);
        if (value > best) { best = value; best_ids = cand; }
    }
    if (score) *score = conf(best_ids);
    return ids_text(best_ids, nullptr);
}

RingOutput RingReader::read(const RsImage& crop) {
    RingOutput out;
    std::vector<float> input = preprocess(crop);
    int len = 0, vocab = 0;
    auto lp = run(*main_, input, len, vocab);
    out.main_text = decode(lp.data(), len, vocab, &out.score);
    out.text = out.main_text;
    if (long_ && static_cast<int>(utf8_chars(out.main_text).size()) >= config_.long_threshold) {
        auto lp2 = run(*long_, input, len, vocab);
        out.text = decode(lp2.data(), len, vocab, &out.score);
        out.routed = true;
    }
    return out;
}

}  // namespace rs
