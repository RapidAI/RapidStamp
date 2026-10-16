#include "rapidstamp.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool read_ppm(const std::string& path, std::vector<std::uint8_t>& rgb, int& width, int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::string magic;
    in >> magic;
    if (magic != "P6") {
        return false;
    }
    in >> width >> height;
    int max_value = 0;
    in >> max_value;
    if (width <= 0 || height <= 0 || max_value != 255) {
        return false;
    }
    in.get();
    rgb.resize(static_cast<size_t>(width * height * 3));
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return in.gcount() == static_cast<std::streamsize>(rgb.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: rapidstamp_demo image.ppm detector.manifest recognizer.manifest\n";
        return 2;
    }
    std::vector<std::uint8_t> rgb;
    int width = 0;
    int height = 0;
    if (!read_ppm(argv[1], rgb, width, height)) {
        std::cerr << "failed to read ppm " << argv[1] << "\n";
        return 1;
    }
    rapidstamp_engine* engine = rapidstamp_create(argv[2], argv[3]);
    if (!engine) {
        std::cerr << rapidstamp_last_error() << "\n";
        return 1;
    }
    rapidstamp_box boxes[16];
    const int n = rapidstamp_infer_rgb(engine, rgb.data(), width, height, width * 3, boxes, 16);
    rapidstamp_destroy(engine);
    if (n < 0) {
        std::cerr << rapidstamp_last_error() << "\n";
        return 1;
    }
    std::cout << "count " << n << "\n";
    for (int i = 0; i < n; ++i) {
        std::cout << boxes[i].x << " " << boxes[i].y << " " << boxes[i].w << " " << boxes[i].h << " "
                  << boxes[i].score << " " << boxes[i].text << " " << boxes[i].text_score << "\n";
    }
    return 0;
}
