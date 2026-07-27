#include <stdio.h>
#include <ccap_c.h>
#include <ccap_utils_c.h>
#include <behema/behema.h>

#include "draw_cortex.h"

// CCAP reference:
// https://github.com/wysaid/CameraCapture/blob/main/examples/desktop/2-capture_grab_c.c

#define CORTEX_WIDTH 200
#define CORTEX_HEIGHT 120
#define CORTEX_NH_RADIUS 2
#define SAMPLE_WINDOW BHM_SAMPLE_WINDOW_MID

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

int clamp(int d, int min, int max) {
   const int t = d < min ? min : d;
   return t > max ? max : t;
}

bhm_error_code_t bhm_context_setup(
    bhm_context2d_t** context,
    bhm_input2d_t** left_eye,
    bhm_input2d_t** right_eye
) {
    bhm_error_code_t error = BHM_ERROR_NONE;

    // Create the context.
    error = bhm_ctx2d_create(context, CORTEX_WIDTH, CORTEX_HEIGHT, CORTEX_NH_RADIUS);
    if (error != BHM_ERROR_NONE) return error;
    printf("Created context\n");

    // Customize cortex properties.
    error = bhm_ctx2d_set_sample_window(*context, SAMPLE_WINDOW);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_set_evol_step(*context, 0x01U);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_set_pulse_mapping(*context, BHM_PULSE_MAPPING_RPROP);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_set_max_syn_count(*context, 24);
    if (error != BHM_ERROR_NONE) return error;
    char touch_file_name[40];
    char inhexc_file_mame[40];
    sprintf(touch_file_name, "./res/%d_%d_touch.pgm", CORTEX_WIDTH, CORTEX_HEIGHT);
    sprintf(inhexc_file_mame, "./res/%d_%d_inhexc.pgm", CORTEX_WIDTH, CORTEX_HEIGHT);
    ctx2d_touch_from_map(*context, touch_file_name);
    ctx2d_inhexc_from_map(*context, inhexc_file_mame);
    printf("Updated context\n");

    // Setup inputs.
    error = bhm_i2d_create(
        left_eye,
        0,
        0,
        (CORTEX_WIDTH / 10) * 3,
        1,
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_FPROP
    );
    if (error != BHM_ERROR_NONE) return error;
    printf("Created left eye\n");

    error = bhm_i2d_create(
        right_eye,
        (CORTEX_WIDTH / 10) * 7,
        0,
        CORTEX_WIDTH,
        1,
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_FPROP
    );
    if (error != BHM_ERROR_NONE) return error;
    printf("Created right eye\n");

    // Add inputs to the context.
    error = bhm_ctx2d_add_input(*context, *left_eye);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_add_input(*context, *right_eye);
    if (error != BHM_ERROR_NONE) return error;
    printf("Added inputs to context\n");

    return error;
}

int main(int argc, char** argv) {
    bhm_error_code_t error;

    // ####################### Behema setup #######################
    // Cortex init.
    bhm_context2d_t* bhm_ctx;
    bhm_input2d_t* left_eye;
    bhm_input2d_t* right_eye;
    error = bhm_context_setup(&bhm_ctx, &left_eye, &right_eye);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing the context %d\n", error);
        return 1;
    }

    bhm_cortex_size_t eye_width = left_eye->x1 - left_eye->x0;

    // ################################## CCAP setup ##################################
    // Create provider.
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

    // Try and set output format.
    ccap_provider_set_property(provider, CCAP_PROPERTY_WIDTH, 640.0);
    ccap_provider_set_property(provider, CCAP_PROPERTY_HEIGHT, 360.0);
    ccap_provider_set_property(provider, CCAP_PROPERTY_PIXEL_FORMAT_OUTPUT, CCAP_PIXEL_FORMAT_BGR24);

    // Start capture
    if (!ccap_provider_start(provider)) return 1;

    bhm_ticks_count_t sampling_bound = SAMPLE_WINDOW - 1;
    bhm_ticks_count_t sample_step = sampling_bound;

    // ################################## Raylib setup ##################################
    InitWindow(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "Camera example"
    );

    const int canvas_width = CORTEX_WIDTH;
    const int canvas_height = CORTEX_HEIGHT;
    RenderTexture2D canvas = LoadRenderTexture(
        canvas_width,
        canvas_height
    );
    SetTextureFilter(canvas.texture, TEXTURE_FILTER_POINT);

    SetTargetFPS(960);

    bhm_render_mode render_mode = BHM_SPIKES;

    // ################################## Main loop ##################################
    while (!WindowShouldClose()) {
        // ################################## Handle user input ##################################
        if (IsKeyPressed(KEY_SPACE)) render_mode = (render_mode + 0x01u) % BHM_RENDER_MODES_COUNT;

        // ################################## Fetch cortex input ##################################
        if (sample_step > sampling_bound) {
            CcapVideoFrame* frame = ccap_provider_grab(provider, 3000);
            if (frame == NULL) {
                printf("ERROR! blank frame grabbed\n");
                break;
            }

            CcapVideoFrameInfo frameInfo;
            if (ccap_video_frame_get_info(frame, &frameInfo)) {
                // Get pixel format string
                char formatStr[64];
                ccap_pixel_format_to_string(frameInfo.pixelFormat, formatStr, sizeof(formatStr));

                // Assuming you already have frameInfo and requested BGR24 format
                uint8_t* pixels = frameInfo.data[0];
                uint32_t stride = frameInfo.stride[0];

                // 1. Jump safely to the start of the current row using the STRIDE
                uint8_t* current_row = pixels + (((int) (frameInfo.height / 2)) * stride);

                // Loop through the destination array size.
                for (int i = 0; i < eye_width; i++) {
                    // 2. Jump to the specific pixel in this row. 
                    // BGR24 means 3 bytes per pixel.
                    int pixel_offset = i * 3;
                    
                    uint8_t b = current_row[pixel_offset + 0];
                    uint8_t g = current_row[pixel_offset + 1];
                    uint8_t r = current_row[pixel_offset + 2];

                    // Ratio between m and n.
                    float src_index_f = (((float) i) + 0.5f) * ((float) stride) / ((float) eye_width) - 0.5f;

                    // Decompose the obtained index to its integer and fractional parts.
                    int src_index_i = (int) src_index_f;
                    float frac = src_index_f - src_index_i;

                    // Pick left and right indices to sample from. The right index is clamped to its max possible value.
                    int l_src_index = clamp(src_index_i, 0, stride);
                    int r_src_index = clamp(src_index_i + 1, 0, stride);

                    int sampled_value = current_row[l_src_index * 3] * (1 - frac) + current_row[r_src_index * 3] * frac;

                    left_eye->values[i] = fmap(sampled_value, 0, 255, 0, sampling_bound);
                    right_eye->values[i] = fmap(sampled_value, 0, 255, 0, sampling_bound);
                }
            }
            ccap_video_frame_release(frame);

            sample_step = 0;
        }

        sample_step++;

        // ################################## Draw ##################################
        BeginTextureMode(canvas);
            ClearBackground(BLACK);
            draw_cortex(
                bhm_ctx->even_cortex,
                render_mode,
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            );
        EndTextureMode();

        BeginDrawing();
            ClearBackground(BLACK);

            // Source rectangle: The entire canvas texture. 
            // The negative height flips the texture upright.
            Rectangle source_rec = { 
                0.0f, 
                0.0f, 
                (float) canvas.texture.width, 
                -(float) canvas.texture.height 
            };

            // Destination rectangle: Where on the screen to draw it and how big.
            // We set it to the screen width and height to scale it.
            Rectangle dest_rec = { 
                0.0f, 
                0.0f, 
                (float) WINDOW_WIDTH, 
                -(float) WINDOW_HEIGHT
            };

            // Origin is the rotation pivot point (top-left is 0,0).
            Vector2 origin = {0.0f, 0.0f};

            // Draw the texture scaled to destRec.
            DrawTexturePro(
                canvas.texture,
                source_rec,
                dest_rec,
                origin,
                0.0f,
                WHITE
            );

            DrawFPS(10, 10);
        EndDrawing();

        // ################################## Tick the cortex ##################################
        ctx2d_tick(bhm_ctx);
    }

    // ################################## Raylib cleanup ##################################
    CloseWindow();

    // ################################## CCAP cleanup ##################################
    ccap_provider_stop(provider);
    ccap_provider_close(provider);
    ccap_provider_destroy(provider);

    // ################################## Behema cleanup ##################################
    bhm_ctx2d_destroy(bhm_ctx);

    return 0;
}
