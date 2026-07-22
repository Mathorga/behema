#include <stdio.h>
#include <ccap_c.h>
#include <ccap_utils_c.h>

// For reference:
// https://github.com/wysaid/CameraCapture/blob/main/examples/desktop/2-capture_grab_c.c

int main() {
    // Create provider
    CcapProvider* provider = ccap_provider_create();
    if (provider == NULL) return -1;
    
    // Find available devices
    CcapDeviceNamesList deviceList;
    int res = ccap_provider_find_device_names_list(provider, &deviceList);
    if (res) {
        printf("Found %zu camera device(s):\n", deviceList.deviceCount);
        for (size_t i = 0; i < deviceList.deviceCount; i++) {
            printf("  %zu: %s\n", i, deviceList.deviceNames[i]);
        }
    }
    
    // Open default camera
    if (!ccap_provider_open(provider, NULL, false)) return 1;

    // Set output format.
    ccap_provider_set_property(provider, CCAP_PROPERTY_WIDTH, 10);
    ccap_provider_set_property(provider, CCAP_PROPERTY_HEIGHT, 10);
    ccap_provider_set_property(provider, CCAP_PROPERTY_PIXEL_FORMAT_OUTPUT, CCAP_PIXEL_FORMAT_BGR24);
    
    // Start capture
    if (!ccap_provider_start(provider)) return 1;

    // Grab a frame
    CcapVideoFrame* frame = ccap_provider_grab(provider, 3000);
    if (frame) {
        CcapVideoFrameInfo frameInfo;
        if (ccap_video_frame_get_info(frame, &frameInfo)) {
            // Get pixel format string
            char formatStr[64];
            ccap_pixel_format_to_string(frameInfo.pixelFormat, formatStr, sizeof(formatStr));
            
            printf(
                "Captured: %dx%d, format=%s\n",
                frameInfo.width,
                frameInfo.height,
                formatStr
            );

            // Assuming you already have frameInfo and requested BGR24 format
            uint8_t* pixels = frameInfo.data[0];
            uint32_t stride = frameInfo.stride[0];

            for (int y = 0; y < frameInfo.height; y++) {
                // 1. Jump safely to the start of the current row using the STRIDE
                uint8_t* current_row = pixels + (y * stride);
                for (int x = 0; x < frameInfo.width; x++) {
                    // 2. Jump to the specific pixel in this row. 
                    // BGR24 means 3 bytes per pixel.
                    int pixel_offset = x * 3;
                    
                    uint8_t b = current_row[pixel_offset + 0];
                    uint8_t g = current_row[pixel_offset + 1];
                    uint8_t r = current_row[pixel_offset + 2];
                    
                    // Safety limit: Only print the first 5x5 pixels so we don't freeze the terminal
                    printf("%c  ", ((r + g + b) / 3) > 200 ? '@' : ' ');
                    
                    // If you actually wanted to process the data (e.g., find the brightest pixel, 
                    // convert to grayscale, etc.), you would do that math here without the printf.
                }
                printf("\n");
            }
        }
        ccap_video_frame_release(frame);
    }
    
    
    ccap_provider_stop(provider);
    ccap_provider_close(provider);
    ccap_provider_destroy(provider);
    return 0;
}
