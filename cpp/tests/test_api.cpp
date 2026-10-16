#include "rapidstamp.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "api test skipped (pass det.onnx rec.onnx charset.txt)\n";
        return 0;
    }
    rapidstamp_engine* engine = rapidstamp_create(argv[1], argv[2], argv[3], nullptr);
    if (!engine) {
        std::cerr << "create failed: " << rapidstamp_last_error() << "\n";
        return 1;
    }
    const int width = 64;
    const int height = 64;
    unsigned char rgb[width * height * 3];
    for (int i = 0; i < width * height; ++i) {
        rgb[i * 3 + 0] = 240;
        rgb[i * 3 + 1] = 240;
        rgb[i * 3 + 2] = 240;
    }
    rapidstamp_seal seals[8];
    int count = rapidstamp_infer(engine, rgb, width, height, width * 3, seals, 8);
    if (count < 0) {
        std::cerr << "infer failed: " << rapidstamp_last_error() << "\n";
        rapidstamp_destroy(engine);
        return 1;
    }
    std::cout << "api ok seals " << count << "\n";
    rapidstamp_destroy(engine);
    return 0;
}
