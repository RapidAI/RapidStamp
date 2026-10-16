#pragma once

#include <cstdint>
#include <vector>

namespace rsp {

struct ImageView {
    const std::uint8_t* data;
    int width;
    int height;
    int stride;
};

struct RingStrip {
    std::vector<float> gray;  // H * W, row-major, 0..1
    int height;
    int width;
};

// Must stay in lockstep with python/src/rapidstamp/geometry.py.
constexpr int kDetSize = 128;
constexpr int kUnwrapH = 32;
constexpr int kUnwrapW = 160;
constexpr float kTextFraction = 0.78f;
constexpr float kTextOuter = 0.86f;
constexpr float kTextInner = 0.58f;
constexpr int kNAngles = 360;
constexpr int kCharSize = 32;
constexpr double kTextEdge = 6.0 * 3.14159265358979323846 / 180.0;
constexpr float kBorderFrac = 0.055f;

void refine_ring(const ImageView& image, float x, float y, float w, float h, float& ox, float& oy, float& ow, float& oh);

std::vector<float> slot_centers(int n_chars, int width = kUnwrapW);
void crop_slot(const std::vector<float>& strip, int height, int width, float center_x, float* dst, int size = kCharSize);

void resize_nchw(const ImageView& image, float* dst, int out_h, int out_w);
RingStrip unwrap_seal(const ImageView& image, float x, float y, float w, float h);

}  // namespace rsp
