#ifndef RAPIDSTAMP_H
#define RAPIDSTAMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAPIDSTAMP_MAX_TEXT 512
#define RAPIDSTAMP_OK 0
#define RAPIDSTAMP_ERR_ARG -1
#define RAPIDSTAMP_ERR_MODEL -2
#define RAPIDSTAMP_ERR_RUNTIME -3

typedef struct rapidstamp_engine rapidstamp_engine;

typedef struct rapidstamp_seal {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float text_score;
    int class_id;
    char text[RAPIDSTAMP_MAX_TEXT];
    char center_text[RAPIDSTAMP_MAX_TEXT];
} rapidstamp_seal;

typedef struct rapidstamp_config {
    float score_thresh;
    float nms_thresh;
    int angle_steps;
    int intra_threads;
} rapidstamp_config;

/* Zeros select the values in preprocess.txt, or the built-in defaults. */
rapidstamp_config rapidstamp_default_config(void);

rapidstamp_engine* rapidstamp_create(const char* det_model_path,
                                     const char* rec_model_path,
                                     const char* charset_path,
                                     const rapidstamp_config* config);

void rapidstamp_destroy(rapidstamp_engine* engine);

/* rgb8 row-major. stride is bytes per row. Returns the number of seals, or <0. */
int rapidstamp_infer(rapidstamp_engine* engine,
                     const uint8_t* rgb,
                     int width,
                     int height,
                     int stride,
                     rapidstamp_seal* out_seals,
                     int max_out);

const char* rapidstamp_class_name(rapidstamp_engine* engine, int class_id);
const char* rapidstamp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
