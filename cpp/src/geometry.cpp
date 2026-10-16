#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {

constexpr double kPi = 3.14159265358979323846;

void sample_bilinear(const RsImage& src, const float* sx, const float* sy, int n, float* out) {
    const int width = src.width;
    const int height = src.height;
    const int channels = src.channels;
    for (int i = 0; i < n; ++i) {
        float x = std::min(std::max(sx[i], 0.f), static_cast<float>(width - 1));
        float y = std::min(std::max(sy[i], 0.f), static_cast<float>(height - 1));
        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
        float wx = x - static_cast<float>(x0);
        float wy = y - static_cast<float>(y0);
        const uint8_t* p00 = src.pixel(x0, y0);
        const uint8_t* p10 = src.pixel(x1, y0);
        const uint8_t* p01 = src.pixel(x0, y1);
        const uint8_t* p11 = src.pixel(x1, y1);
        for (int c = 0; c < channels; ++c) {
            float top = p00[c] * (1.f - wx) + p10[c] * wx;
            float bot = p01[c] * (1.f - wx) + p11[c] * wx;
            out[i * channels + c] = top * (1.f - wy) + bot * wy;
        }
    }
}

RsImage from_float(const float* sampled, int width, int height, int channels) {
    RsImage out;
    out.width = width;
    out.height = height;
    out.channels = channels;
    out.data.resize(static_cast<size_t>(width) * height * channels);
    for (size_t i = 0; i < out.data.size(); ++i) {
        float value = sampled[i] + 0.5f;
        if (value < 0.f) value = 0.f;
        if (value > 255.f) value = 255.f;
        out.data[i] = static_cast<uint8_t>(value);
    }
    return out;
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

float iou(const RsDet& a, const RsDet& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

}  // namespace

RsImage resize_bilinear(const RsImage& src, int out_w, int out_h) {
    if (src.width == out_w && src.height == out_h) return src;
    std::vector<float> sx(static_cast<size_t>(out_w) * out_h);
    std::vector<float> sy(sx.size());
    for (int y = 0; y < out_h; ++y) {
        double yy = (static_cast<double>(y) + 0.5) * (static_cast<double>(src.height) / out_h) - 0.5;
        for (int x = 0; x < out_w; ++x) {
            double xx = (static_cast<double>(x) + 0.5) * (static_cast<double>(src.width) / out_w) - 0.5;
            size_t index = static_cast<size_t>(y) * out_w + x;
            sx[index] = static_cast<float>(xx);
            sy[index] = static_cast<float>(yy);
        }
    }
    std::vector<float> sampled(sx.size() * src.channels);
    sample_bilinear(src, sx.data(), sy.data(), static_cast<int>(sx.size()), sampled.data());
    return from_float(sampled.data(), out_w, out_h, src.channels);
}

RsLetterbox letterbox_image(const RsImage& src, int size, int pad_value) {
    RsLetterbox result;
    float scale = static_cast<float>(size) / static_cast<float>(std::max(src.height, src.width));
    int new_w = std::min(size, std::max(1, static_cast<int>(std::lround(src.width * scale))));
    int new_h = std::min(size, std::max(1, static_cast<int>(std::lround(src.height * scale))));
    RsImage resized = resize_bilinear(src, new_w, new_h);
    result.image.width = size;
    result.image.height = size;
    result.image.channels = src.channels;
    result.image.data.assign(static_cast<size_t>(size) * size * src.channels, static_cast<uint8_t>(pad_value));
    result.pad_x = (size - new_w) / 2;
    result.pad_y = (size - new_h) / 2;
    result.scale = scale;
    for (int y = 0; y < new_h; ++y) {
        std::copy(resized.pixel(0, y), resized.pixel(0, y) + new_w * src.channels,
                  result.image.pixel(result.pad_x, result.pad_y + y));
    }
    return result;
}

RsImage pad_to_square(const RsImage& src, uint8_t pad_value) {
    int side = std::max(src.width, src.height);
    if (src.width == side && src.height == side) return src;
    RsImage out;
    out.width = side;
    out.height = side;
    out.channels = src.channels;
    out.data.assign(static_cast<size_t>(side) * side * src.channels, pad_value);
    int left = (side - src.width) / 2;
    int top = (side - src.height) / 2;
    for (int y = 0; y < src.height; ++y) {
        std::copy(src.pixel(0, y), src.pixel(0, y) + src.width * src.channels, out.pixel(left, top + y));
    }
    return out;
}

RsImage polar_unwrap(const RsImage& src, int out_w, int out_h, float r_in, float r_out, float start_deg,
                     float span_deg) {
    const double cx = (src.width - 1) / 2.0;
    const double cy = (src.height - 1) / 2.0;
    const double rad_x = (src.width - 1) / 2.0;
    const double rad_y = (src.height - 1) / 2.0;
    std::vector<float> sx(static_cast<size_t>(out_w) * out_h);
    std::vector<float> sy(sx.size());
    for (int y = 0; y < out_h; ++y) {
        double t = r_out + (r_in - r_out) * ((static_cast<double>(y) + 0.5) / out_h);
        for (int x = 0; x < out_w; ++x) {
            double frac = (static_cast<double>(x) + 0.5) / out_w;
            double angle = (start_deg + frac * span_deg) * kPi / 180.0;
            size_t index = static_cast<size_t>(y) * out_w + x;
            sx[index] = static_cast<float>(cx + rad_x * t * std::cos(angle));
            sy[index] = static_cast<float>(cy + rad_y * t * std::sin(angle));
        }
    }
    std::vector<float> sampled(sx.size() * src.channels);
    sample_bilinear(src, sx.data(), sy.data(), static_cast<int>(sx.size()), sampled.data());
    return from_float(sampled.data(), out_w, out_h, src.channels);
}

RsImage grid_unroll(const RsImage& src, int rows, int cols, int out_h, int out_w) {
    rows = std::max(1, rows);
    cols = std::max(1, cols);
    int cell_w = std::max(1, out_w / (rows * cols));
    RsImage strip;
    strip.width = cell_w * rows * cols;
    strip.height = out_h;
    strip.channels = src.channels;
    strip.data.resize(static_cast<size_t>(strip.width) * strip.height * src.channels);
    int col_index = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col, ++col_index) {
            int y1 = static_cast<int>(std::lround(row * src.height / static_cast<double>(rows)));
            int y2 = static_cast<int>(std::lround((row + 1) * src.height / static_cast<double>(rows)));
            int x1 = static_cast<int>(std::lround(col * src.width / static_cast<double>(cols)));
            int x2 = static_cast<int>(std::lround((col + 1) * src.width / static_cast<double>(cols)));
            y1 = std::min(std::max(y1, 0), src.height);
            y2 = std::min(std::max(y2, y1 + 1), src.height);
            x1 = std::min(std::max(x1, 0), src.width);
            x2 = std::min(std::max(x2, x1 + 1), src.width);
            RsImage cell;
            cell.width = x2 - x1;
            cell.height = y2 - y1;
            cell.channels = src.channels;
            cell.data.resize(static_cast<size_t>(cell.width) * cell.height * src.channels);
            for (int y = 0; y < cell.height; ++y) {
                std::copy(src.pixel(x1, y1 + y), src.pixel(x1, y1 + y) + cell.width * src.channels, cell.pixel(0, y));
            }
            RsImage scaled = resize_bilinear(cell, cell_w, out_h);
            for (int y = 0; y < out_h; ++y) {
                std::copy(scaled.pixel(0, y), scaled.pixel(0, y) + cell_w * src.channels,
                          strip.pixel(col_index * cell_w, y));
            }
        }
    }
    if (strip.width != out_w) return resize_bilinear(strip, out_w, out_h);
    return strip;
}

RsImage hline_band(const RsImage& src, int out_h, int out_w) {
    int top = std::max(0, static_cast<int>(std::lround(src.height * 0.22)));
    int bot = std::min(src.height, std::max(top + 1, static_cast<int>(std::lround(src.height * 0.78))));
    RsImage band;
    band.width = src.width;
    band.height = bot - top;
    band.channels = src.channels;
    band.data.resize(static_cast<size_t>(band.width) * band.height * src.channels);
    for (int y = 0; y < band.height; ++y) {
        std::copy(src.pixel(0, top + y), src.pixel(0, top + y) + src.width * src.channels, band.pixel(0, y));
    }
    return resize_bilinear(band, out_w, out_h);
}

void gray01(const RsImage& src, std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(src.width) * src.height);
    for (int i = 0; i < src.width * src.height; ++i) {
        const uint8_t* p = src.data.data() + static_cast<size_t>(i) * src.channels;
        float gray = src.channels == 1 ? p[0] : (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
        dst[i] = gray / 255.f;
    }
}

std::vector<RsDet> decode_detections(const float* pred, int cells, int num_classes, float score_thresh,
                                     float nms_thresh, float box_limit) {
    const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(cells))));
    std::vector<RsDet> candidates;
    for (int i = 0; i < cells; ++i) {
        const float* row = pred + static_cast<size_t>(i) * (5 + num_classes);
        float tx = row[0];
        float ty = row[1];
        float tw = std::min(std::max(row[2], -box_limit), box_limit);
        float th = std::min(std::max(row[3], -box_limit), box_limit);
        float cx = (static_cast<float>(i % grid) + sigmoid(tx)) / static_cast<float>(grid);
        float cy = (static_cast<float>(i / grid) + sigmoid(ty)) / static_cast<float>(grid);
        float bw = std::exp(tw) / static_cast<float>(grid);
        float bh = std::exp(th) / static_cast<float>(grid);
        int best = 0;
        float best_logit = row[5];
        for (int c = 1; c < num_classes; ++c) {
            if (row[5 + c] > best_logit) {
                best_logit = row[5 + c];
                best = c;
            }
        }
        float max_logit = best_logit;
        float sum = 0.f;
        for (int c = 0; c < num_classes; ++c) sum += std::exp(row[5 + c] - max_logit);
        float class_prob = std::exp(best_logit - max_logit) / std::max(sum, 1e-6f);
        float score = sigmoid(row[4]) * class_prob;
        if (score < score_thresh) continue;
        RsDet det;
        det.x1 = cx - bw / 2.f;
        det.y1 = cy - bh / 2.f;
        det.x2 = cx + bw / 2.f;
        det.y2 = cy + bh / 2.f;
        det.score = score;
        det.class_id = best;
        candidates.push_back(det);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const RsDet& a, const RsDet& b) { return a.score > b.score; });
    std::vector<RsDet> kept;
    std::vector<int> removed(candidates.size(), 0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (removed[j] || candidates[j].class_id != candidates[i].class_id) continue;
            if (iou(candidates[i], candidates[j]) > nms_thresh) removed[j] = 1;
        }
    }
    return kept;
}

bool estimate_ring_start(const RsImage& src, float r_in, float r_out, float& start_deg, float& gap_deg) {
    // Same 96×96 histogram as python rapidstamp.geometry.estimate_ring_start.
    RsImage small = resize_bilinear(src, 96, 96);
    constexpr int kBins = 72;
    int counts[kBins] = {};
    int ink_n = 0;
    const double cx = (small.width - 1) / 2.0;
    const double cy = (small.height - 1) / 2.0;
    for (int y = 0; y < small.height; ++y) {
        for (int x = 0; x < small.width; ++x) {
            double dx = (static_cast<double>(x) - cx) / std::max(cx, 1e-6);
            double dy = (static_cast<double>(y) - cy) / std::max(cy, 1e-6);
            double radius = std::sqrt(dx * dx + dy * dy);
            if (radius < r_in || radius > r_out + 0.02) continue;
            const uint8_t* px = small.pixel(x, y);
            int red = px[0];
            int green = small.channels > 1 ? px[1] : px[0];
            int blue = small.channels > 2 ? px[2] : px[0];
            bool ink = (red > 120 && red > green + 30 && red > blue + 30) ||
                       (blue > 100 && blue > red + 20 && blue > green + 10) || (red < 85 && green < 85 && blue < 85);
            if (!ink) continue;
            double angle = std::atan2(dy, dx) * 180.0 / 3.14159265358979323846;
            if (angle < 0) angle += 360.0;
            int bin = static_cast<int>(angle / 360.0 * kBins);
            if (bin < 0) bin = 0;
            if (bin >= kBins) bin = kBins - 1;
            counts[bin] += 1;
            ink_n += 1;
        }
    }
    gap_deg = 0.f;
    if (ink_n < 25) return false;
    int best = 0;
    int best_end = 0;
    int run = 0;
    for (int index = 0; index < kBins * 2; ++index) {
        if (counts[index % kBins] == 0) {
            run += 1;
            if (run > best && run <= kBins) {
                best = run;
                best_end = index;
            }
        } else {
            run = 0;
        }
    }
    gap_deg = static_cast<float>(best * (360.0 / kBins));
    if (best < 8) return false;
    int start_bin = (best_end + 1) % kBins;
    start_deg = static_cast<float>(start_bin * (360.0 / kBins));
    return true;
}

bool load_key_values(const std::string& path, std::vector<std::pair<std::string, std::string>>& out) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string key;
        stream >> key;
        std::string rest;
        std::getline(stream, rest);
        if (!rest.empty() && rest[0] == ' ') rest.erase(rest.begin());
        out.emplace_back(key, rest);
    }
    return true;
}
