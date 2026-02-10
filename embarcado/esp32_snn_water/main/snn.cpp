#include "snn.h"
#include "weights.h"
#include "model_config.h"
#include <math.h>

struct LIF {
    float v;
};

static LIF hidden[HIDDEN_SIZE];

static inline float norm(float x, float m, float s) {
    return (x - m) / s;
}

void snn_predict(float in_raw[3], float out[3]) {

    float x[3];
    x[0] = norm(in_raw[0], PH_MEAN, PH_STD);
    x[1] = norm(in_raw[1], TEMP_MEAN, TEMP_STD);
    x[2] = norm(in_raw[2], TURB_MEAN, TURB_STD);

    for (int h = 0; h < HIDDEN_SIZE; h++)
        hidden[h].v = 0;

    for (int o = 0; o < OUTPUT_SIZE; o++)
        out[o] = 0;

    for (int t = 0; t < T_STEPS; t++) {

        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float acc = B1[h];
            for (int i = 0; i < INPUT_SIZE; i++)
                acc += W1[h*INPUT_SIZE + i] * x[i];

            hidden[h].v = hidden[h].v * BETA + acc;

            if (hidden[h].v > VTH)
                hidden[h].v = VRESET;
        }

        for (int o = 0; o < OUTPUT_SIZE; o++) {
            float acc = B2[o];
            for (int h = 0; h < HIDDEN_SIZE; h++)
                acc += W2[o*HIDDEN_SIZE + h] * hidden[h].v;
            out[o] += acc;
        }
    }

    for (int o = 0; o < OUTPUT_SIZE; o++)
        out[o] /= T_STEPS;
}
