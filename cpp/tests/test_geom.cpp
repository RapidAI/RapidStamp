#include "geometry.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool load_ints(std::istream& input, std::vector<int>& values, int count) {
    values.clear();
    values.reserve(count);
    for (int i = 0; i < count; ++i) {
        int value = 0;
        if (!(input >> value)) return false;
        values.push_back(value);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "testdata/unwrap_golden.txt";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        return 1;
    }
    int width = 0, height = 0, channels = 0, out_w = 0, out_h = 0;
    input >> width >> height >> channels >> out_w >> out_h;
    float r_in = 0, r_out = 0, start = 0, span = 0;
    input >> r_in >> r_out >> start >> span;
    std::vector<int> pixels;
    std::vector<int> expected;
    if (!load_ints(input, pixels, width * height * channels) ||
        !load_ints(input, expected, out_w * out_h * channels)) {
        std::cerr << "golden file is truncated\n";
        return 1;
    }
    RsImage image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.data.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) image.data[i] = static_cast<uint8_t>(pixels[i]);
    RsImage unwrapped = polar_unwrap(image, out_w, out_h, r_in, r_out, start, span);
    int mismatches = 0;
    int worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        int delta = std::abs(static_cast<int>(unwrapped.data[i]) - expected[i]);
        worst = std::max(worst, delta);
        if (delta > 1) ++mismatches;
    }
    if (mismatches) {
        std::cerr << "unwrap mismatch " << mismatches << " worst " << worst << "\n";
        return 1;
    }

    // A mark at the top of a circle (phi = 270) lands near column (270-start)/span.
    RsImage dot;
    dot.width = 32;
    dot.height = 32;
    dot.channels = 3;
    dot.data.assign(32 * 32 * 3, 0);
    dot.pixel(16, 4)[0] = 255;
    RsImage strip = polar_unwrap(dot, 64, 8, 0.2f, 1.f, 140.f, 360.f);
    int peak = 0;
    int peak_value = -1;
    for (int x = 0; x < strip.width; ++x) {
        int value = 0;
        for (int y = 0; y < strip.height; ++y) value += strip.pixel(x, y)[0];
        if (value > peak_value) {
            peak_value = value;
            peak = x;
        }
    }
    int expected_col = static_cast<int>(std::lround((270.0 - 140.0) / 360.0 * 64.0));
    if (std::abs(peak - expected_col) > 2 || peak_value <= 0) {
        std::cerr << "angle peak " << peak << " expected " << expected_col << " value " << peak_value << "\n";
        return 1;
    }
    std::cout << "geometry ok worst_delta " << worst << "\n";
    return 0;
}
