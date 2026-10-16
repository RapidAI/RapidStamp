#ifndef RAPIDSTAMP_H
#define RAPIDSTAMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RAPIDSTAMP_STATIC)
#define RAPIDSTAMP_API
#elif defined(_WIN32)
#ifdef RAPIDSTAMP_BUILD
#define RAPIDSTAMP_API __declspec(dllexport)
#else
#define RAPIDSTAMP_API __declspec(dllimport)
#endif
#else
#define RAPIDSTAMP_API __attribute__((visibility("default")))
#endif

typedef struct rapidstamp_engine rapidstamp_engine;

typedef struct rapidstamp_box {
    float x;
    float y;
    float w;
    float h;
    float score;
    char text[512];
    float text_score;
} rapidstamp_box;

/* manifest paths end in .manifest. The sibling .bin holds float32 weights. */
RAPIDSTAMP_API rapidstamp_engine* rapidstamp_create(const char* det_manifest, const char* rec_manifest);
RAPIDSTAMP_API void rapidstamp_destroy(rapidstamp_engine* engine);
RAPIDSTAMP_API void rapidstamp_set_thresholds(rapidstamp_engine* engine, float det_thresh, float nms_thresh);

/* rgb is tightly or strided packed RGB, row-major. Returns the number of
 * seals written to out_boxes, or -1 on error (see rapidstamp_last_error). */
RAPIDSTAMP_API int rapidstamp_infer_rgb(
    rapidstamp_engine* engine,
    const uint8_t* rgb,
    int width,
    int height,
    int stride_bytes,
    rapidstamp_box* out_boxes,
    int max_boxes);

RAPIDSTAMP_API const char* rapidstamp_last_error(void);
RAPIDSTAMP_API const char* rapidstamp_version(void);

#ifdef __cplusplus
}
#endif

#endif
