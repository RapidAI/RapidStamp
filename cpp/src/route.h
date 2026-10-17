#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

// Class -> "ppocr" or "ctc". A bare class name means ppocr.
// A reader token other than those two is ignored. '#' starts a comment.
// Python rapidstamp.labels.load_rec_route parses the same lines.
inline std::unordered_map<std::string, std::string> parse_rec_route(const std::string& text) {
    std::unordered_map<std::string, std::string> route;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream row(line);
        std::string name;
        std::string reader;
        if (!(row >> name)) continue;
        if (!(row >> reader)) reader = "ppocr";
        if (reader != "ppocr" && reader != "ctc") continue;
        route[name] = reader;
    }
    return route;
}
