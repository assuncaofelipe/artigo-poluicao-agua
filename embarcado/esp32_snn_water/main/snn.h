#pragma once

#include "model_config.h"
#include "weights.h"
#include <cmath>
#include <cstring>

// Leaky Integrate-and-Fire Neuron Network
class SpikingNeuralNetwork {
private:
    float v_hidden[MODEL_HIDDEN_SIZE];  // Membrane potentials
    float s_hidden[MODEL_HIDDEN_SIZE];  // Spikes
    
public:
    SpikingNeuralNetwork() {
        reset();
    }
    
    void reset() {
        memset(v_hidden, 0, sizeof(v_hidden));
        memset(s_hidden, 0, sizeof(s_hidden));
    }
    
    void normalize_input(float* input) {
        // Z-score normalization using training stats from model_config.h
        input[0] = (input[0] - NORM_MEAN_PH) / (NORM_STD_PH + 1e-7f);
        input[1] = (input[1] - NORM_MEAN_TEMP) / (NORM_STD_TEMP + 1e-7f);
        input[2] = (input[2] - NORM_MEAN_TURB) / (NORM_STD_TURB + 1e-7f);
    }
    
    void step(const float* input, float* output) {
        // Forward propagation: input -> hidden
        for(int h = 0; h < MODEL_HIDDEN_SIZE; h++) {
            float I_syn = 0.0f;
            for(int i = 0; i < MODEL_INPUT_SIZE; i++) {
                I_syn += W1[h][i] * input[i];
            }
            I_syn += B1[h];
            
            // LIF neuron: v(t+1) = beta * v(t) + I_syn
            v_hidden[h] = SNN_BETA * v_hidden[h] + I_syn;
            
            // Spike generation
            s_hidden[h] = (v_hidden[h] >= SNN_THRESHOLD) ? 1.0f : 0.0f;
            
            // Reset after spike
            if(s_hidden[h] > 0.5f) {
                v_hidden[h] = 0.0f;
            }
        }
        
        // Readout: hidden -> output
        for(int o = 0; o < MODEL_OUTPUT_SIZE; o++) {
            float sum = 0.0f;
            for(int h = 0; h < MODEL_HIDDEN_SIZE; h++) {
                sum += W2[o][h] * s_hidden[h];
            }
            output[o] = sum + B2[o];
        }
    }
    
    void infer(const float* input_raw, float* output) {
        float input[MODEL_INPUT_SIZE];
        memcpy(input, input_raw, sizeof(float) * MODEL_INPUT_SIZE);
        
        reset();
        normalize_input(input);
        
        // Simulate for SNN_TIME_STEPS timesteps
        for(int t = 0; t < SNN_TIME_STEPS; t++) {
            step(input, output);
        }
    }
};

void snn_predict(float in[3], float out[1]);
