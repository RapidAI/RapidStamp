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

/* rec_model_path + charset_path is the CTC recognizer.
   If seal_rec_ppocr.onnx, charset_ppocr.txt and rec_route.txt sit beside
   charset_path, classes marked ppocr in that file use the second model.
   Missing siblings leave every class on CTC. The argument list stays four. */
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

/* PARSeq ring reader (whole crop, lexicon rescoring, length routing).
   release_dir holds route.json, reader_*.onnx, charset.txt and the name list;
   see python/src/rapidstamp/ring_reader.py. names_path NULL and
   long_threshold < 0 keep the route.json values. If <model dir>/ring_reader/
   route.json exists beside charset_path, rapidstamp_create loads it too and
   rapidstamp_infer fills seal.text from it for the classes it lists
   (default: every class except square_personal). */
typedef struct rapidstamp_ring_reader rapidstamp_ring_reader;
rapidstamp_ring_reader* rapidstamp_ring_reader_create(const char* release_dir,
                                                      int intra_threads,
                                                      const char* names_path,
                                                      int long_threshold);
void rapidstamp_ring_reader_destroy(rapidstamp_ring_reader* reader);
/* Crop is rgb8. Writes UTF-8 text. routed is 1 when the long model answered. */
int rapidstamp_ring_read(rapidstamp_ring_reader* reader,
                         const uint8_t* rgb,
                         int width,
                         int height,
                         int stride,
                         char* text,
                         int text_cap,
                         float* score,
                         int* routed);

const char* rapidstamp_class_name(rapidstamp_engine* engine, int class_id);
const char* rapidstamp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
