#pragma once

#include "constants.h"

#include <cstdint>
#include <string>
#include <vector>

struct RsImage {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;

    uint8_t* pixel(int x, int y) { return data.data() + (static_cast<size_t>(y) * width + x) * channels; }
    const uint8_t* pixel(int x, int y) const {
        return data.data() + (static_cast<size_t>(y) * width + x) * channels;
    }
};

struct RsLetterbox {
    RsImage image;
    float scale = 1.f;
    int pad_x = 0;
    int pad_y = 0;
};

struct RsDet {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    int class_id = 0;
};

RsImage resize_bilinear(const RsImage& src, int out_w, int out_h);
RsLetterbox letterbox_image(const RsImage& src, int size, int pad_value);
RsImage pad_to_square(const RsImage& src, uint8_t pad_value);
RsImage polar_unwrap(const RsImage& src, int out_w, int out_h, float r_in, float r_out, float start_deg,
                     float span_deg);
RsImage grid_unroll(const RsImage& src, int rows, int cols, int out_h, int out_w);
RsImage hline_band(const RsImage& src, int out_h, int out_w);
// Largest ink gap on the ring. False when the gap is under 40 degrees.
bool estimate_ring_start(const RsImage& src, float r_in, float r_out, float& start_deg, float& gap_deg);

void gray01(const RsImage& src, std::vector<float>& dst);

std::vector<RsDet> decode_detections(const float* pred, int cells, int num_classes, float score_thresh,
                                     float nms_thresh, float box_limit);

bool load_key_values(const std::string& path, std::vector<std::pair<std::string, std::string>>& out);
