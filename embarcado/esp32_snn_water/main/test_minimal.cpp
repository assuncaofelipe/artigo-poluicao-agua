#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" void app_main(void) {
    printf("\n\n");
    printf("==== Firmware Minimal ====\n");
    printf("Startup complete!\n\n");
    
    // Simple task
    int counter = 0;
    while(1) {
        printf("Loop: %d\n", counter++);
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
