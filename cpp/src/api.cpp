#include "rapidstamp.h"

#include "geometry.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_error;

void set_error(const std::string& message) { g_error = message; }

struct Box {
    float x, y, w, h, score;
};

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

float box_iou(const Box& a, const Box& b) {
    const float ax2 = a.x + a.w;
    const float ay2 = a.y + a.h;
    const float bx2 = b.x + b.w;
    const float by2 = b.y + b.h;
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = std::max(0.f, ix2 - ix1);
    const float ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni = a.w * a.h + b.w * b.h - inter;
    if (uni <= 0.f) {
        return 0.f;
    }
    return inter / uni;
}

std::vector<Box> nms(std::vector<Box> boxes, float thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> kept;
    for (const Box& box : boxes) {
        bool ok = true;
        for (const Box& other : kept) {
            if (box_iou(box, other) >= thresh) {
                ok = false;
                break;
            }
        }
        if (ok) {
            kept.push_back(box);
        }
    }
    return kept;
}

std::vector<Box> decode(const rsp::Tensor& logits, int orig_w, int orig_h, float thresh, float nms_thresh, int det_size, int grid) {
    const float sx = static_cast<float>(orig_w) / static_cast<float>(det_size);
    const float sy = static_cast<float>(orig_h) / static_cast<float>(det_size);
    const int g = grid;
    std::vector<Box> boxes;
    for (int row = 0; row < g; ++row) {
        for (int col = 0; col < g; ++col) {
            const int hw = g * g;
            const float obj = sigmoid(logits.data[0 * hw + row * g + col]);
            if (obj < thresh) {
                continue;
            }
            const float tx = sigmoid(logits.data[1 * hw + row * g + col]);
            const float ty = sigmoid(logits.data[2 * hw + row * g + col]);
            const float tw = sigmoid(logits.data[3 * hw + row * g + col]);
            const float th = sigmoid(logits.data[4 * hw + row * g + col]);
            const float cx = tx * static_cast<float>(det_size);
            const float cy = ty * static_cast<float>(det_size);
            const float bw = tw * static_cast<float>(det_size);
            const float bh = th * static_cast<float>(det_size);
            Box box;
            box.x = (cx - bw * 0.5f) * sx;
            box.y = (cy - bh * 0.5f) * sy;
            box.w = bw * sx;
            box.h = bh * sy;
            box.score = obj;
            boxes.push_back(box);
        }
    }
    return nms(boxes, nms_thresh);
}

std::vector<std::string> utf8_chars(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t n = 1;
        if ((lead & 0x80u) == 0) {
            n = 1;
        } else if ((lead & 0xE0u) == 0xC0u) {
            n = 2;
        } else if ((lead & 0xF0u) == 0xE0u) {
            n = 3;
        } else if ((lead & 0xF8u) == 0xF0u) {
            n = 4;
        }
        if (i + n > text.size()) {
            n = 1;
        }
        out.push_back(text.substr(i, n));
        i += n;
    }
    return out;
}

rsp::Tensor slot_logits(const rsp::Model& rec, const rsp::RingStrip& strip, int n_chars) {
    const std::vector<float> centers = rsp::slot_centers(n_chars, strip.width);
    std::vector<float> batch(static_cast<size_t>(n_chars * rsp::kCharSize * rsp::kCharSize));
    for (int i = 0; i < n_chars; ++i) {
        rsp::crop_slot(
            strip.gray,
            strip.height,
            strip.width,
            centers[static_cast<size_t>(i)],
            batch.data() + static_cast<size_t>(i * rsp::kCharSize * rsp::kCharSize));
    }
    return rsp::run_model(rec, batch.data(), n_chars, 1, rsp::kCharSize, rsp::kCharSize);
}

void classify_strip(const rsp::Model& rec, const rsp::RingStrip& strip, std::string& text, float& score) {
    text.clear();
    score = 0.f;
    if (!rec.words.empty()) {
        std::unordered_map<std::string, int> index;
        for (size_t i = 0; i < rec.classes.size(); ++i) {
            index.emplace(rec.classes[i], static_cast<int>(i));
        }
        std::vector<int> lengths;
        for (const std::string& word : rec.words) {
            const int n = static_cast<int>(utf8_chars(word).size());
            if (std::find(lengths.begin(), lengths.end(), n) == lengths.end()) {
                lengths.push_back(n);
            }
        }
        std::sort(lengths.begin(), lengths.end());
        double best = -1.0e300;
        for (int n_chars : lengths) {
            const rsp::Tensor logits = slot_logits(rec, strip, n_chars);
            const int classes = logits.d[1];
            std::vector<double> logp(static_cast<size_t>(n_chars * classes));
            for (int i = 0; i < n_chars; ++i) {
                const float* row = logits.data.data() + static_cast<size_t>(i * classes);
                float m = row[0];
                for (int c = 1; c < classes; ++c) {
                    m = std::max(m, row[c]);
                }
                double sum = 0.0;
                for (int c = 0; c < classes; ++c) {
                    sum += std::exp(static_cast<double>(row[c] - m));
                }
                const double logz = std::log(sum);
                for (int c = 0; c < classes; ++c) {
                    logp[static_cast<size_t>(i * classes + c)] = static_cast<double>(row[c] - m) - logz;
                }
            }
            for (const std::string& word : rec.words) {
                const std::vector<std::string> chars = utf8_chars(word);
                if (static_cast<int>(chars.size()) != n_chars) {
                    continue;
                }
                double total = 0.0;
                bool ok = true;
                for (int i = 0; i < n_chars; ++i) {
                    const auto found = index.find(chars[static_cast<size_t>(i)]);
                    if (found == index.end() || found->second >= classes) {
                        ok = false;
                        break;
                    }
                    total += logp[static_cast<size_t>(i * classes + found->second)];
                }
                if (ok && total > best) {
                    best = total;
                    text = word;
                }
            }
        }
        if (!text.empty()) {
            score = static_cast<float>(std::exp(best / static_cast<double>(utf8_chars(text).size())));
        }
        return;
    }
    score = -1.f;
    for (int n_chars : {5, 7}) {
        const rsp::Tensor logits = slot_logits(rec, strip, n_chars);
        const int classes = logits.d[1];
        double conf = 0.0;
        std::string candidate;
        for (int i = 0; i < n_chars; ++i) {
            const float* row = logits.data.data() + static_cast<size_t>(i * classes);
            int arg = 0;
            float best_logit = row[0];
            for (int c = 1; c < classes; ++c) {
                if (row[c] > best_logit) {
                    best_logit = row[c];
                    arg = c;
                }
            }
            float sum = 0.f;
            for (int c = 0; c < classes; ++c) {
                sum += std::exp(row[c] - best_logit);
            }
            conf += 1.0 / static_cast<double>(sum);
            if (arg < static_cast<int>(rec.classes.size())) {
                candidate += rec.classes[static_cast<size_t>(arg)];
            }
        }
        const float mean = static_cast<float>(conf / n_chars);
        if (mean > score) {
            score = mean;
            text = std::move(candidate);
        }
    }
}

}  // namespace

struct rapidstamp_engine {
    rsp::Model det;
    rsp::Model rec;
    float det_thresh;
    float nms_thresh;
};

extern "C" {

const char* rapidstamp_version(void) { return "0.1.0"; }

const char* rapidstamp_last_error(void) { return g_error.c_str(); }

rapidstamp_engine* rapidstamp_create(const char* det_manifest, const char* rec_manifest) {
    g_error.clear();
    try {
        auto* engine = new rapidstamp_engine();
        engine->det = rsp::load_model(det_manifest);
        engine->rec = rsp::load_model(rec_manifest);
        if (engine->det.kind != "detector" || engine->rec.kind != "recognizer") {
            delete engine;
            set_error("manifest kinds must be detector and recognizer");
            return nullptr;
        }
        engine->det_thresh = engine->det.meta_float("det_thresh", 0.5f);
        engine->nms_thresh = engine->det.meta_float("nms_thresh", 0.45f);
        return engine;
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return nullptr;
    }
}

void rapidstamp_destroy(rapidstamp_engine* engine) { delete engine; }

void rapidstamp_set_thresholds(rapidstamp_engine* engine, float det_thresh, float nms_thresh) {
    if (!engine) {
        return;
    }
    engine->det_thresh = det_thresh;
    engine->nms_thresh = nms_thresh;
}

int rapidstamp_infer_rgb(
    rapidstamp_engine* engine,
    const std::uint8_t* rgb,
    int width,
    int height,
    int stride_bytes,
    rapidstamp_box* out_boxes,
    int max_boxes) {
    g_error.clear();
    if (!engine || !rgb || !out_boxes || width <= 0 || height <= 0 || max_boxes <= 0) {
        set_error("invalid infer arguments");
        return -1;
    }
    if (stride_bytes < width * 3) {
        set_error("stride is smaller than width * 3");
        return -1;
    }
    try {
        rsp::ImageView image{rgb, width, height, stride_bytes};
        std::vector<float> det_in(static_cast<size_t>(3 * rsp::kDetSize * rsp::kDetSize));
        rsp::resize_nchw(image, det_in.data(), engine->det.input_h, engine->det.input_w);
        rsp::Tensor det_out = rsp::run_model(engine->det, det_in.data(), 1, 3, engine->det.input_h, engine->det.input_w);
        const int grid = engine->det.meta_int("grid", 8);
        std::vector<Box> boxes = decode(det_out, width, height, engine->det_thresh, engine->nms_thresh, engine->det.input_h, grid);
        if (static_cast<int>(boxes.size()) > max_boxes) {
            boxes.resize(static_cast<size_t>(max_boxes));
        }
        for (size_t i = 0; i < boxes.size(); ++i) {
            const Box& box = boxes[i];
            float rx = box.x;
            float ry = box.y;
            float rw = box.w;
            float rh = box.h;
            rsp::refine_ring(image, box.x, box.y, box.w, box.h, rx, ry, rw, rh);
            rsp::RingStrip strip = rsp::unwrap_seal(image, rx, ry, rw, rh);
            std::string text;
            float text_score = 0.f;
            classify_strip(engine->rec, strip, text, text_score);
            rapidstamp_box& out = out_boxes[i];
            out.x = rx;
            out.y = ry;
            out.w = rw;
            out.h = rh;
            out.score = box.score;
            out.text_score = text_score;
            std::memset(out.text, 0, sizeof(out.text));
            const size_t n = std::min(text.size(), sizeof(out.text) - 1);
            std::memcpy(out.text, text.data(), n);
        }
        return static_cast<int>(boxes.size());
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

}  // extern "C"
