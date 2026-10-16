#include "rapidstamp.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool read_ppm(const std::string& path, std::vector<uint8_t>& rgb, int& width, int& height) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::string magic;
    input >> magic;
    if (magic != "P6") return false;
    int max_value = 0;
    input >> width >> height >> max_value;
    if (max_value != 255 || width <= 0 || height <= 0) return false;
    input.get();
    rgb.resize(static_cast<size_t>(width) * height * 3);
    input.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return input.good() || input.eof();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: rapidstamp_demo det.onnx rec.onnx charset.txt image.ppm\n";
        return 2;
    }
    rapidstamp_engine* engine = rapidstamp_create(argv[1], argv[2], argv[3], nullptr);
    if (!engine) {
        std::cerr << rapidstamp_last_error() << "\n";
        return 1;
    }
    std::vector<uint8_t> rgb;
    int width = 0, height = 0;
    if (!read_ppm(argv[4], rgb, width, height)) {
        std::cerr << "failed to read ppm\n";
        rapidstamp_destroy(engine);
        return 1;
    }
    rapidstamp_seal seals[32];
    int count = rapidstamp_infer(engine, rgb.data(), width, height, width * 3, seals, 32);
    if (count < 0) {
        std::cerr << rapidstamp_last_error() << "\n";
        rapidstamp_destroy(engine);
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        const rapidstamp_seal& seal = seals[i];
        std::cout << rapidstamp_class_name(engine, seal.class_id) << " score=" << seal.score
                  << " text=" << seal.text << " text_score=" << seal.text_score
                  << " bbox=" << seal.x1 << "," << seal.y1 << "," << seal.x2 << "," << seal.y2 << "\n";
    }
    rapidstamp_destroy(engine);
    return 0;
}
