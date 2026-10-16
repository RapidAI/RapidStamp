#include "runtime.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_manifest_and_bin() {
    std::ofstream manifest("rsp_test_conv.manifest");
    manifest << "rapidstamp 1\n";
    manifest << "kind detector\n";
    manifest << "input 1 1 2 2\n";
    manifest << "weight_count 2\n";
    manifest << "op conv 1 1 1 1 1 1 0 0 1\n";
    manifest << "op relu\n";
    std::vector<float> weights = {2.f, 0.5f};
    std::ofstream bin("rsp_test_conv.bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(float)));
}

bool close(float a, float b) { return std::fabs(a - b) < 1e-5f; }

}  // namespace

int main() {
    try {
        write_manifest_and_bin();
        const rsp::Model model = rsp::load_model("rsp_test_conv.manifest");
        const float input[4] = {1.f, -2.f, 3.f, 0.5f};
        const rsp::Tensor out = rsp::run_model(model, input, 1, 1, 2, 2);
        const float expected[4] = {2.5f, 0.f, 6.5f, 1.5f};
        if (out.rank != 4 || out.d[1] != 1 || out.d[2] != 2 || out.d[3] != 2) {
            std::cerr << "unexpected shape\n";
            return 1;
        }
        for (int i = 0; i < 4; ++i) {
            if (!close(out.data[static_cast<size_t>(i)], expected[i])) {
                std::cerr << "mismatch at " << i << " got " << out.data[static_cast<size_t>(i)] << "\n";
                return 1;
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
