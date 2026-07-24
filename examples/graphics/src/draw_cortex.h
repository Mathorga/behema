#include <raylib.h>
#include <behema/behema.h>

bhm_error_code_t draw_cortex(
    bhm_cortex2d_t* cortex,
    int window_width,
    int window_height
) {
    const int cell_width = 4;
    const int cell_height = 4;

    const int starting_x = window_width - cortex->width * cell_width;
    const int starting_y = 0;

    // ClearBackground(BLACK);

    for (bhm_cortex_size_t j = 0; j < cortex->height; j++) {
        for (bhm_cortex_size_t i = 0; i < cortex->width; i++) {

            bhm_neuron_t* currentNeuron = &(cortex->neurons[BHM_IDX2D(i, j, cortex->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold + (float) (currentNeuron->pulse));

            bool fired = currentNeuron->pulse_mask & 0x01U;

            Color neuron_color = BLACK;

            if (fired) {
                neuron_color = WHITE;
            } else {
                if (neuronValue < 0) {
                    neuron_color = (Color) {
                        0x00,
                        127,
                        255,
                        31 - 31 * neuronValue
                    };
                } else {
                    neuron_color = (Color) {
                        0x00,
                        127,
                        255,
                        31 + 224 * neuronValue
                    };
                }
            }

            DrawPixel(i, j, neuron_color);
        }
    }

    // Draw cortex info.
    const int text_padding = 8;
    const int font_size = 20;

    const char* width_text = TextFormat(
        "width: %i",
        cortex->width
    );
    const char* height_text = TextFormat(
        "height: %i",
        cortex->height
    );
    const int width_text_size = MeasureText(width_text, font_size);
    const int height_text_size = MeasureText(height_text, font_size);

    DrawText(
        width_text,
        window_width - width_text_size - text_padding,
        window_height - (text_padding + font_size + 20),
        font_size,
        RAYWHITE
    );
    DrawText(
        height_text,
        window_width - height_text_size - text_padding,
        window_height - (text_padding + font_size),
        font_size,
        RAYWHITE
    );

    return BHM_ERROR_NONE;
}