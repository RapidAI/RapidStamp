#include "runtime.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    in.seekg(0);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: rapidstamp_parity manifest input.bin N C H W output.bin\n";
        return 2;
    }
    try {
        const rsp::Model model = rsp::load_model(argv[1]);
        const std::vector<float> input = read_f32(argv[2]);
        const int n = std::stoi(argv[3]);
        const int c = std::stoi(argv[4]);
        const int h = std::stoi(argv[5]);
        const int w = std::stoi(argv[6]);
        if (static_cast<int>(input.size()) != n * c * h * w) {
            std::cerr << "input size mismatch\n";
            return 1;
        }
        const rsp::Tensor out = rsp::run_model(model, input.data(), n, c, h, w);
        std::ofstream file(argv[7], std::ios::binary);
        file.write(reinterpret_cast<const char*>(out.data.data()), static_cast<std::streamsize>(out.data.size() * sizeof(float)));
        return file ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
