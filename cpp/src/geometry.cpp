#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

namespace rsp {
namespace {

void sample(const ImageView& image, float x, float y, float& r, float& g, float& b) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = static_cast<float>(x - x0);
    const float wy = static_cast<float>(y - y0);
    auto at = [&](int xx, int yy, float& pr, float& pg, float& pb) {
        if (xx < 0 || yy < 0 || xx >= image.width || yy >= image.height) {
            pr = pg = pb = 255.f;
            return;
        }
        const std::uint8_t* p = image.data + yy * image.stride + xx * 3;
        pr = p[0];
        pg = p[1];
        pb = p[2];
    };
    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    at(x0, y0, r00, g00, b00);
    at(x1, y0, r10, g10, b10);
    at(x0, y1, r01, g01, b01);
    at(x1, y1, r11, g11, b11);
    const float wa = (1.f - wx) * (1.f - wy);
    const float wb = wx * (1.f - wy);
    const float wc = (1.f - wx) * wy;
    const float wd = wx * wy;
    r = r00 * wa + r10 * wb + r01 * wc + r11 * wd;
    g = g00 * wa + g10 * wb + g01 * wc + g11 * wd;
    b = b00 * wa + b10 * wb + b01 * wc + b11 * wd;
}

std::vector<float> smooth_circular(const std::vector<float>& values, int k) {
    if (k % 2 == 0) {
        ++k;
    }
    const int n = static_cast<int>(values.size());
    const int pad = k / 2;
    std::vector<float> ext(static_cast<size_t>(n + 2 * pad));
    for (int i = 0; i < pad; ++i) {
        ext[static_cast<size_t>(i)] = values[static_cast<size_t>(n - pad + i)];
    }
    std::copy(values.begin(), values.end(), ext.begin() + pad);
    for (int i = 0; i < pad; ++i) {
        ext[static_cast<size_t>(pad + n + i)] = values[static_cast<size_t>(i)];
    }
    std::vector<float> out(static_cast<size_t>(n), 0.f);
    const float scale = 1.f / static_cast<float>(k);
    for (int i = 0; i < n; ++i) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) {
            sum += ext[static_cast<size_t>(i + j)];
        }
        out[static_cast<size_t>(i)] = sum * scale;
    }
    return out;
}

int estimate_text_start(const std::vector<float>& ink, float text_fraction) {
    const std::vector<float> smoothed = smooth_circular(ink, 7);
    const int n = static_cast<int>(smoothed.size());
    int gap_bins = static_cast<int>(std::lround(n * (1.0 - static_cast<double>(text_fraction))));
    gap_bins = std::min(std::max(gap_bins, 1), n - 1);
    const int text_bins = n - gap_bins;
    std::vector<double> csum(static_cast<size_t>(2 * n + 1), 0.0);
    for (int i = 0; i < 2 * n; ++i) {
        csum[static_cast<size_t>(i + 1)] = csum[static_cast<size_t>(i)] + smoothed[static_cast<size_t>(i % n)];
    }
    int best = 0;
    double best_sum = csum[static_cast<size_t>(text_bins)] - csum[0];
    for (int i = 1; i < n; ++i) {
        const double sum = csum[static_cast<size_t>(i + text_bins)] - csum[static_cast<size_t>(i)];
        if (sum > best_sum) {
            best_sum = sum;
            best = i;
        }
    }
    return best;
}

std::vector<float> angular_ink(const ImageView& image, float cx, float cy, float r_in, float r_out) {
    constexpr int kRadii = 8;
    std::vector<float> ink(kNAngles, 0.f);
    for (int i = 0; i < kNAngles; ++i) {
        const double angle = (static_cast<double>(i) + 0.5) * (2.0 * kPi / kNAngles);
        float acc = 0.f;
        for (int k = 0; k < kRadii; ++k) {
            const float t = (kRadii == 1) ? 0.f : static_cast<float>(k) / static_cast<float>(kRadii - 1);
            const float radius = r_in + (r_out - r_in) * t;
            const float x = cx + static_cast<float>(std::cos(angle)) * radius;
            const float y = cy + static_cast<float>(std::sin(angle)) * radius;
            float r, g, b;
            sample(image, x, y, r, g, b);
            acc += ((255.f - r) + (255.f - g) + (255.f - b)) / 3.f;
        }
        ink[static_cast<size_t>(i)] = acc / static_cast<float>(kRadii);
    }
    return ink;
}

float ring_ink(const ImageView& image, float cx, float cy, float radius) {
    constexpr int kAngles = 72;
    double acc = 0.0;
    for (int i = 0; i < kAngles; ++i) {
        const double angle = (static_cast<double>(i) + 0.5) * (2.0 * kPi / kAngles);
        const float x = cx + static_cast<float>(std::cos(angle)) * radius;
        const float y = cy + static_cast<float>(std::sin(angle)) * radius;
        float r, g, b;
        sample(image, x, y, r, g, b);
        acc += ((255.0 - r) + (255.0 - g) + (255.0 - b)) / 3.0;
    }
    return static_cast<float>(acc / kAngles);
}

void search_ring(
    const ImageView& image,
    float cx,
    float cy,
    float radius,
    const std::vector<double>& dcxs,
    const std::vector<double>& dcys,
    const std::vector<double>& scales,
    float& best_x,
    float& best_y,
    float& best_r) {
    double best = -1.0e300;
    bool any = false;
    for (double dcx : dcxs) {
        for (double dcy : dcys) {
            for (double scale : scales) {
                const float px = cx + static_cast<float>(dcx) * radius;
                const float py = cy + static_cast<float>(dcy) * radius;
                const float pr = radius * static_cast<float>(scale);
                const double score = static_cast<double>(ring_ink(image, px, py, pr * (1.f - kBorderFrac * 0.5f))) -
                                     static_cast<double>(ring_ink(image, px, py, pr * 0.35f));
                if (!any || score > best) {
                    any = true;
                    best = score;
                    best_x = px;
                    best_y = py;
                    best_r = pr;
                }
            }
        }
    }
}

std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return out;
}

}  // namespace

void refine_ring(const ImageView& image, float x, float y, float w, float h, float& ox, float& oy, float& ow, float& oh) {
    const float cx = x + w * 0.5f;
    const float cy = y + h * 0.5f;
    const float radius = 0.5f * std::min(w, h);
    if (radius < 4.f) {
        ox = cx - radius;
        oy = cy - radius;
        ow = radius * 2.f;
        oh = radius * 2.f;
        return;
    }
    float bx = cx;
    float by = cy;
    float br = radius;
    search_ring(image, cx, cy, radius, linspace(-0.40, 0.40, 9), linspace(-0.40, 0.40, 9), linspace(0.80, 1.20, 5), bx, by, br);
    const double base_dx = static_cast<double>(bx - cx) / static_cast<double>(radius);
    const double base_dy = static_cast<double>(by - cy) / static_cast<double>(radius);
    const double base_sc = static_cast<double>(br) / static_cast<double>(radius);
    float fx = bx;
    float fy = by;
    float fr = br;
    search_ring(
        image,
        cx,
        cy,
        radius,
        linspace(base_dx - 0.10, base_dx + 0.10, 9),
        linspace(base_dy - 0.10, base_dy + 0.10, 9),
        linspace(base_sc - 0.08, base_sc + 0.08, 5),
        fx,
        fy,
        fr);
    ox = fx - fr;
    oy = fy - fr;
    ow = fr * 2.f;
    oh = fr * 2.f;
}

std::vector<float> slot_centers(int n_chars, int width) {
    const double arc = 2.0 * kPi * static_cast<double>(kTextFraction);
    const double edge_frac = kTextEdge / arc;
    const double usable = 1.0 - 2.0 * edge_frac;
    std::vector<float> centers(static_cast<size_t>(n_chars));
    for (int i = 0; i < n_chars; ++i) {
        const double frac = edge_frac + (static_cast<double>(i) + 0.5) * usable / n_chars;
        centers[static_cast<size_t>(i)] = static_cast<float>(frac * width - 0.5);
    }
    return centers;
}

void crop_slot(const std::vector<float>& strip, int height, int width, float center_x, float* dst, int size) {
    const int x0 = static_cast<int>(std::lround(static_cast<double>(center_x) - size / 2.0));
    std::fill(dst, dst + height * size, 1.f);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < size; ++col) {
            const int src_x = x0 + col;
            if (src_x < 0 || src_x >= width) {
                continue;
            }
            dst[row * size + col] = strip[static_cast<size_t>(row * width + src_x)];
        }
    }
}

void resize_nchw(const ImageView& image, float* dst, int out_h, int out_w) {
    if (image.height == out_h && image.width == out_w) {
        for (int y = 0; y < out_h; ++y) {
            const std::uint8_t* row = image.data + y * image.stride;
            for (int x = 0; x < out_w; ++x) {
                dst[0 * out_h * out_w + y * out_w + x] = row[x * 3 + 0] / 255.f;
                dst[1 * out_h * out_w + y * out_w + x] = row[x * 3 + 1] / 255.f;
                dst[2 * out_h * out_w + y * out_w + x] = row[x * 3 + 2] / 255.f;
            }
        }
        return;
    }
    for (int y = 0; y < out_h; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) * (static_cast<float>(image.height) / out_h) - 0.5f;
        for (int x = 0; x < out_w; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * (static_cast<float>(image.width) / out_w) - 0.5f;
            float r, g, b;
            sample(image, sx, sy, r, g, b);
            dst[0 * out_h * out_w + y * out_w + x] = r / 255.f;
            dst[1 * out_h * out_w + y * out_w + x] = g / 255.f;
            dst[2 * out_h * out_w + y * out_w + x] = b / 255.f;
        }
    }
}

RingStrip unwrap_seal(const ImageView& image, float x, float y, float w, float h) {
    const float cx = x + w * 0.5f;
    const float cy = y + h * 0.5f;
    const float radius = 0.5f * std::min(w, h);
    const float r_out = radius * kTextOuter;
    const float r_in = radius * kTextInner;
    const std::vector<float> ink = angular_ink(image, cx, cy, r_in, r_out);
    const int start_idx = estimate_text_start(ink, kTextFraction);
    const double start = start_idx * (2.0 * kPi / kNAngles);
    const double arc = 2.0 * kPi * static_cast<double>(kTextFraction);
    RingStrip strip;
    strip.height = kUnwrapH;
    strip.width = kUnwrapW;
    strip.gray.assign(static_cast<size_t>(kUnwrapH * kUnwrapW), 0.f);
    for (int row = 0; row < kUnwrapH; ++row) {
        const float rt = (static_cast<float>(row) + 0.5f) / static_cast<float>(kUnwrapH);
        const float rad = r_out + (r_in - r_out) * rt;
        for (int col = 0; col < kUnwrapW; ++col) {
            const double ang = start + ((static_cast<double>(col) + 0.5) / kUnwrapW) * arc;
            const float sx = cx + static_cast<float>(std::cos(ang)) * rad;
            const float sy = cy + static_cast<float>(std::sin(ang)) * rad;
            float r, g, b;
            sample(image, sx, sy, r, g, b);
            strip.gray[static_cast<size_t>(row * kUnwrapW + col)] = (r + g + b) / (3.f * 255.f);
        }
    }
    return strip;
}

}  // namespace rsp
