// Batch driver for the ring reader: one crop (binary PPM) per line.
// usage: rapidstamp_ring_eval release_dir names|- threshold list.tsv out.tsv [threads]
//   list.tsv: id<TAB>path.ppm     out.tsv: id<TAB>text<TAB>routed<TAB>ms
#include "rapidstamp.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool read_ppm(const std::string& path, std::vector<uint8_t>& rgb, int& width, int& height) {
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
    return true;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: rapidstamp_ring_eval release_dir names|- threshold list.tsv out.tsv [threads]\n";
        return 2;
    }
    std::string names = argv[2];
    int threads = argc > 6 ? std::stoi(argv[6]) : 0;
    rapidstamp_ring_reader* reader =
        rapidstamp_ring_reader_create(argv[1], threads, names == "-" ? nullptr : names.c_str(), std::stoi(argv[3]));
    if (!reader) {
        std::cerr << rapidstamp_last_error() << "\n";
        return 1;
    }
    std::ifstream list(argv[4]);
    std::ofstream out(argv[5], std::ios::binary);
    std::string line;
    std::vector<uint8_t> rgb;
    char text[RAPIDSTAMP_MAX_TEXT];
    int n = 0;
    while (std::getline(list, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string id = line.substr(0, tab), path = line.substr(tab + 1);
        int w = 0, h = 0;
        if (!read_ppm(path, rgb, w, h)) {
            std::cerr << "bad ppm " << path << "\n";
            continue;
        }
        float score = 0;
        int routed = 0;
        auto t0 = std::chrono::steady_clock::now();
        int rc = rapidstamp_ring_read(reader, rgb.data(), w, h, w * 3, text, sizeof(text), &score, &routed);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rc != RAPIDSTAMP_OK) {
            std::cerr << id << ": " << rapidstamp_last_error() << "\n";
            text[0] = 0;
        }
        out << id << '\t' << text << '\t' << routed << '\t' << ms << '\n';
        if (++n % 500 == 0) std::cerr << n << "\n";
    }
    rapidstamp_ring_reader_destroy(reader);
    return 0;
}
