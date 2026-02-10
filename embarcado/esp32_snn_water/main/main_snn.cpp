/**
 * @file main_simple.cpp
 * @brief Firmware simples para testar ESP32 sem SNN
 */

#include <stdio.h>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static int sensor_tick = 0;

void simple_task(void *arg) {
    printf("\n\n");
    printf("=================================================\n");
    printf("   ESP32 Simple Water Quality Test\n");
    printf("=================================================\n\n");
    
    while(1) {
        sensor_tick++;
        
        // Simular leitura dinâmica de sensores
        float ph_base = 6.8f;
        float temp_base = 28.5f;
        float turb_base = 2.1f;
        
        // Sine wave simulation
        float sine_wave = 0.5f * sinf(2.0f * 3.14159f * sensor_tick / 50.0f);
        
        float ph = ph_base + 0.2f * sine_wave;           // 6.6 - 7.0
        float temperature = temp_base + 1.5f * sine_wave; // 27.0 - 30.0
        float turbidity = turb_base + 0.5f * sine_wave;  // 1.6 - 2.6
        
        printf("[SENSOR] pH=%.2f Temp=%.2f Turb=%.2f | tick=%d\n",
               ph, temperature, turbidity, sensor_tick);
        
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

extern "C" void app_main(void) {
    printf("\n");
    printf("=================================================\n");
    printf("   ESP32 Simple Water Quality Test\n");
    printf("=================================================\n");
    
    // Start simple task on core 1
    xTaskCreatePinnedToCore(
        simple_task,
        "simple_task",
        4096,
        NULL,
        2,
        NULL,
        1
    );
    
    printf("Simple task started.\n\n");
}
