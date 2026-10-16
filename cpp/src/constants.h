#pragma once

// Keep these defaults aligned with python/src/rapidstamp/labels.py.
// preprocess.txt overrides them at runtime.

constexpr int kDetSize = 320;
constexpr int kGrid = 20;
constexpr int kRecH = 32;
constexpr int kRecW = 384;
constexpr float kUnwrapRIn = 0.72f;
constexpr float kUnwrapROut = 0.99f;
constexpr float kUnwrapStartDeg = 140.f;
constexpr float kUnwrapSpanDeg = 360.f;
constexpr int kPadValue = 114;
constexpr float kScoreThresh = 0.25f;
constexpr float kNmsThresh = 0.45f;
constexpr int kAngleSteps = 24;
constexpr float kBoxLogitLimit = 4.f;
