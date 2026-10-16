#include "rapidstamp.h"

#include "geometry.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

thread_local std::string g_error;

void set_error(const std::string& message) { g_error = message; }

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

std::string parent_dir(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

#ifdef _WIN32
std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring out(static_cast<size_t>(std::max(size, 1)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, out.data(), size);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}
#endif

struct RuntimeConfig {
    int det_size = kDetSize;
    int rec_h = kRecH;
    int rec_w = kRecW;
    float r_in = kUnwrapRIn;
    float r_out = kUnwrapROut;
    float start_deg = kUnwrapStartDeg;
    float span_deg = kUnwrapSpanDeg;
    int pad_value = kPadValue;
    float score_thresh = kScoreThresh;
    float nms_thresh = kNmsThresh;
    int angle_steps = kAngleSteps;
    float box_limit = kBoxLogitLimit;
    std::vector<std::pair<std::string, std::string>> layouts;
};

RuntimeConfig load_config(const std::string& path) {
    RuntimeConfig config;
    std::vector<std::pair<std::string, std::string>> rows;
    if (!load_key_values(path, rows)) return config;
    for (const auto& row : rows) {
        if (row.first == "layout") {
            std::istringstream stream(row.second);
            std::string name, layout;
            stream >> name >> layout;
            config.layouts.emplace_back(name, layout);
        } else if (row.first == "det_size") config.det_size = std::stoi(row.second);
        else if (row.first == "rec_h") config.rec_h = std::stoi(row.second);
        else if (row.first == "rec_w") config.rec_w = std::stoi(row.second);
        else if (row.first == "r_in") config.r_in = std::stof(row.second);
        else if (row.first == "r_out") config.r_out = std::stof(row.second);
        else if (row.first == "start_deg") config.start_deg = std::stof(row.second);
        else if (row.first == "span_deg") config.span_deg = std::stof(row.second);
        else if (row.first == "pad_value") config.pad_value = std::stoi(row.second);
        else if (row.first == "score_thresh") config.score_thresh = std::stof(row.second);
        else if (row.first == "nms_thresh") config.nms_thresh = std::stof(row.second);
        else if (row.first == "angle_steps") config.angle_steps = std::stoi(row.second);
        else if (row.first == "box_logit_limit") config.box_limit = std::stof(row.second);
    }
    return config;
}

std::string layout_of(const RuntimeConfig& config, const std::string& name) {
    for (const auto& item : config.layouts) {
        if (item.first == name) return item.second;
    }
    return "circle";
}

void copy_text(char* dst, int cap, const std::string& text) {
    if (cap <= 0) return;
    std::snprintf(dst, static_cast<size_t>(cap), "%s", text.c_str());
}

}  // namespace

struct rapidstamp_engine {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rapidstamp"};
    std::unique_ptr<Ort::Session> det;
    std::unique_ptr<Ort::Session> rec;
    std::vector<std::string> charset;
    std::vector<std::string> classes;
    RuntimeConfig config;
    int threads = 4;
    std::mutex mutex;

    std::pair<std::string, float> recognize(const RsImage& strip);
    std::pair<std::string, float> read_layout(const RsImage& crop, const std::string& layout);
};

std::pair<std::string, float> ctc_greedy(const float* logits, int time_steps, int classes,
                                        const std::vector<std::string>& charset) {
    std::string text;
    float score_sum = 0.f;
    int previous = 0;
    for (int t = 0; t < time_steps; ++t) {
        const float* row = logits + static_cast<size_t>(t) * classes;
        int best = 0;
        float best_logit = row[0];
        for (int c = 1; c < classes; ++c) {
            if (row[c] > best_logit) {
                best_logit = row[c];
                best = c;
            }
        }
        float max_logit = best_logit;
        float sum = 0.f;
        for (int c = 0; c < classes; ++c) sum += std::exp(row[c] - max_logit);
        float prob = std::exp(best_logit - max_logit) / std::max(sum, 1e-6f);
        if (best != 0 && best != previous) {
            int index = best - 1;
            if (index >= 0 && index < static_cast<int>(charset.size())) {
                text += charset[index];
                score_sum += prob;
            }
        }
        previous = best;
    }
    // Divide by the fixed length so a short confident fragment does not beat a full title.
    return {text, time_steps > 0 ? score_sum / static_cast<float>(time_steps) : 0.f};
}

std::pair<std::string, float> rapidstamp_engine::recognize(const RsImage& strip_in) {
    RsImage strip = strip_in;
    if (strip.width != config.rec_w || strip.height != config.rec_h) {
        strip = resize_bilinear(strip, config.rec_w, config.rec_h);
    }
    std::vector<float> gray;
    gray01(strip, gray);
    std::array<int64_t, 4> shape{1, 1, config.rec_h, config.rec_w};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, gray.data(), gray.size(), shape.data(), shape.size());
    const char* in_names[] = {"image"};
    const char* out_names[] = {"logits"};
    auto outputs = rec->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto dims = info.GetShape();
    int time_steps = static_cast<int>(dims.size() == 3 ? dims[1] : dims[0]);
    int nclass = static_cast<int>(dims.back());
    const float* logits = outputs[0].GetTensorData<float>();
    return ctc_greedy(logits, time_steps, nclass, charset);
}

std::pair<std::string, float> rapidstamp_engine::read_layout(const RsImage& crop, const std::string& layout) {
    if (layout == "hline") return recognize(hline_band(crop, config.rec_h, config.rec_w));
    if (layout == "grid") {
        std::pair<std::string, float> best{"", -1.f};
        const int grids[4][2] = {{2, 1}, {3, 1}, {4, 1}, {2, 2}};
        for (const auto& item : grids) {
            auto decoded = recognize(grid_unroll(crop, item[0], item[1], config.rec_h, config.rec_w));
            if (decoded.second > best.second) best = decoded;
        }
        return best;
    }
    RsImage source = layout == "circle" ? pad_to_square(crop, 255) : crop;
    std::pair<std::string, float> best{"", -1.f};
    int steps = std::max(1, config.angle_steps);
    float best_start = config.start_deg;
    for (int step = 0; step < steps; ++step) {
        float start = step * (360.f / static_cast<float>(steps));
        auto decoded = recognize(polar_unwrap(source, config.rec_w, config.rec_h, config.r_in, config.r_out, start,
                                              config.span_deg));
        if (decoded.second > best.second) {
            best = decoded;
            best_start = start;
        }
    }
    auto canonical = recognize(polar_unwrap(source, config.rec_w, config.rec_h, config.r_in, config.r_out,
                                            config.start_deg, config.span_deg));
    if (canonical.second > best.second) {
        best = canonical;
        best_start = config.start_deg;
    }
    // Same extra bands as python/src/rapidstamp/pipeline.py. Only the winning
    // angle and its neighbors are re-read, so this stays a small extra cost.
    const float extra_bands[2][2] = {{0.64f, 0.94f}, {0.78f, 0.995f}};
    float step_deg = 360.f / static_cast<float>(steps);
    const float refine_starts[4] = {best_start - step_deg, best_start, best_start + step_deg, config.start_deg};
    for (float start : refine_starts) {
        for (const auto& band : extra_bands) {
            auto decoded = recognize(polar_unwrap(source, config.rec_w, config.rec_h, band[0], band[1], start,
                                                  config.span_deg));
            if (decoded.second > best.second) best = decoded;
        }
    }
    return best;
}

rapidstamp_config rapidstamp_default_config(void) {
    rapidstamp_config config{};
    config.score_thresh = 0.f;
    config.nms_thresh = 0.f;
    config.angle_steps = 0;
    config.intra_threads = 4;
    return config;
}

rapidstamp_engine* rapidstamp_create(const char* det_model_path, const char* rec_model_path, const char* charset_path,
                                     const rapidstamp_config* config) {
    if (!det_model_path || !rec_model_path || !charset_path) {
        set_error("null model path");
        return nullptr;
    }
    try {
        auto* engine = new rapidstamp_engine();
        engine->charset = read_lines(charset_path);
        if (engine->charset.empty()) {
            set_error("charset is empty");
            delete engine;
            return nullptr;
        }
        std::string dir = parent_dir(charset_path);
        engine->classes = read_lines(dir + "/classes.txt");
        engine->config = load_config(dir + "/preprocess.txt");
        if (config) {
            if (config->score_thresh > 0.f) engine->config.score_thresh = config->score_thresh;
            if (config->nms_thresh > 0.f) engine->config.nms_thresh = config->nms_thresh;
            if (config->angle_steps > 0) engine->config.angle_steps = config->angle_steps;
            if (config->intra_threads > 0) engine->threads = config->intra_threads;
        }
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(engine->threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        engine->det = std::make_unique<Ort::Session>(engine->env, widen(det_model_path).c_str(), options);
        engine->rec = std::make_unique<Ort::Session>(engine->env, widen(rec_model_path).c_str(), options);
#else
        engine->det = std::make_unique<Ort::Session>(engine->env, det_model_path, options);
        engine->rec = std::make_unique<Ort::Session>(engine->env, rec_model_path, options);
#endif
        g_error.clear();
        return engine;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return nullptr;
    }
}

void rapidstamp_destroy(rapidstamp_engine* engine) { delete engine; }

int rapidstamp_infer(rapidstamp_engine* engine, const uint8_t* rgb, int width, int height, int stride,
                     rapidstamp_seal* out_seals, int max_out) {
    if (!engine || !rgb || width <= 1 || height <= 1 || stride < width * 3 || !out_seals || max_out <= 0) {
        set_error("bad infer arguments");
        return RAPIDSTAMP_ERR_ARG;
    }
    try {
        std::lock_guard<std::mutex> lock(engine->mutex);
        RsImage image;
        image.width = width;
        image.height = height;
        image.channels = 3;
        image.data.resize(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            std::memcpy(image.pixel(0, y), rgb + static_cast<size_t>(y) * stride, static_cast<size_t>(width) * 3);
        }
        RsLetterbox boxed = letterbox_image(image, engine->config.det_size, engine->config.pad_value);
        std::vector<float> input(static_cast<size_t>(3) * boxed.image.width * boxed.image.height);
        for (int y = 0; y < boxed.image.height; ++y) {
            for (int x = 0; x < boxed.image.width; ++x) {
                const uint8_t* p = boxed.image.pixel(x, y);
                size_t plane = static_cast<size_t>(boxed.image.width) * boxed.image.height;
                size_t offset = static_cast<size_t>(y) * boxed.image.width + x;
                input[offset] = p[0] / 255.f;
                input[plane + offset] = p[1] / 255.f;
                input[2 * plane + offset] = p[2] / 255.f;
            }
        }
        std::array<int64_t, 4> shape{1, 3, boxed.image.height, boxed.image.width};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor =
            Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        const char* in_names[] = {"image"};
        const char* out_names[] = {"pred"};
        auto outputs = engine->det->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int cells = static_cast<int>(dims[1]);
        int channels = static_cast<int>(dims[2]);
        int num_classes = channels - 5;
        const float* pred = outputs[0].GetTensorData<float>();
        auto dets = decode_detections(pred, cells, num_classes, engine->config.score_thresh, engine->config.nms_thresh,
                                      engine->config.box_limit);
        int written = 0;
        for (const auto& det : dets) {
            if (written >= max_out) break;
            float x1 = (det.x1 * engine->config.det_size - boxed.pad_x) / boxed.scale;
            float y1 = (det.y1 * engine->config.det_size - boxed.pad_y) / boxed.scale;
            float x2 = (det.x2 * engine->config.det_size - boxed.pad_x) / boxed.scale;
            float y2 = (det.y2 * engine->config.det_size - boxed.pad_y) / boxed.scale;
            int ix1 = std::max(0, static_cast<int>(std::floor(x1)));
            int iy1 = std::max(0, static_cast<int>(std::floor(y1)));
            int ix2 = std::min(width, static_cast<int>(std::ceil(x2)));
            int iy2 = std::min(height, static_cast<int>(std::ceil(y2)));
            if (ix2 - ix1 < 8 || iy2 - iy1 < 8) continue;
            RsImage crop;
            crop.width = ix2 - ix1;
            crop.height = iy2 - iy1;
            crop.channels = 3;
            crop.data.resize(static_cast<size_t>(crop.width) * crop.height * 3);
            for (int y = 0; y < crop.height; ++y) {
                std::copy(image.pixel(ix1, iy1 + y), image.pixel(ix1, iy1 + y) + crop.width * 3, crop.pixel(0, y));
            }
            std::string name = det.class_id >= 0 && det.class_id < static_cast<int>(engine->classes.size())
                                   ? engine->classes[det.class_id]
                                   : "";
            auto decoded = engine->read_layout(crop, layout_of(engine->config, name));
            rapidstamp_seal& slot = out_seals[written++];
            std::memset(&slot, 0, sizeof(slot));
            slot.x1 = x1;
            slot.y1 = y1;
            slot.x2 = x2;
            slot.y2 = y2;
            slot.score = det.score;
            slot.text_score = decoded.second;
            slot.class_id = det.class_id;
            copy_text(slot.text, RAPIDSTAMP_MAX_TEXT, decoded.first);
            if (name == "invoice") {
                int cy1 = static_cast<int>(crop.height * 0.36);
                int cy2 = static_cast<int>(crop.height * 0.68);
                int cx1 = static_cast<int>(crop.width * 0.16);
                int cx2 = static_cast<int>(crop.width * 0.84);
                if (cy2 > cy1 && cx2 > cx1) {
                    RsImage band;
                    band.width = cx2 - cx1;
                    band.height = cy2 - cy1;
                    band.channels = 3;
                    band.data.resize(static_cast<size_t>(band.width) * band.height * 3);
                    for (int y = 0; y < band.height; ++y) {
                        std::copy(crop.pixel(cx1, cy1 + y), crop.pixel(cx1, cy1 + y) + band.width * 3, band.pixel(0, y));
                    }
                    auto center = engine->recognize(resize_bilinear(band, engine->config.rec_w, engine->config.rec_h));
                    std::string digits;
                    for (char ch : center.first) {
                        if (ch >= '0' && ch <= '9') digits.push_back(ch);
                    }
                    if (digits.size() >= 8) copy_text(slot.center_text, RAPIDSTAMP_MAX_TEXT, digits);
                }
            }
        }
        return written;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return RAPIDSTAMP_ERR_RUNTIME;
    }
}

const char* rapidstamp_class_name(rapidstamp_engine* engine, int class_id) {
    if (!engine || class_id < 0 || class_id >= static_cast<int>(engine->classes.size())) return "";
    return engine->classes[class_id].c_str();
}

const char* rapidstamp_last_error(void) { return g_error.c_str(); }
