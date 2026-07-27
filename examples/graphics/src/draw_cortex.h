#include <raylib.h>
#include <behema/behema.h>

typedef enum {
    BHM_SPIKES = 0x00u,
    BHM_SYNAPSES = 0x01u,
    BHM_RENDER_MODES_COUNT
} bhm_render_mode;

bhm_error_code_t draw_cortex(
    bhm_cortex2d_t* cortex,
    bhm_render_mode render_mode,
    int window_width,
    int window_height
) {
    const int cell_width = 4;
    const int cell_height = 4;

    const int starting_x = window_width - cortex->width * cell_width;
    const int starting_y = 0;

    float nh_count = ((float) BHM_NH_COUNT_2D(BHM_NH_DIAM_2D(cortex->nh_radius)));
    // ClearBackground(BLACK);

    for (bhm_cortex_size_t j = 0; j < cortex->height; j++) {
        for (bhm_cortex_size_t i = 0; i < cortex->width; i++) {

            bhm_neuron_t* current_neuron = &(cortex->neurons[BHM_IDX2D(i, j, cortex->width)]);

            float neuron_value = ((float) current_neuron->value) / ((float) cortex->fire_threshold + (float) (current_neuron->pulse));

            bool fired = current_neuron->pulse_mask & 0x01U;

            Color neuron_color = BLACK;

            float syn_count_value = ((float) current_neuron->syn_count) / nh_count;
            switch (render_mode) {
                case BHM_SPIKES:
                    if (fired) {
                        neuron_color = WHITE;
                    } else {
                        if (neuron_value < 0) {
                            neuron_color = (Color) {
                                0x00,
                                127,
                                255,
                                31 - 31 * neuron_value
                            };
                        } else {
                            neuron_color = (Color) {
                                0x00,
                                127,
                                255,
                                31 + 224 * neuron_value
                            };
                        }
                    }
                    break;
                case BHM_SYNAPSES:
                    bhm_syn_count_t exc_count = 0;
                    bhm_syn_count_t inh_count = 0;
                    for (bhm_cortex_size_t k = 0; k < BHM_NH_DIAM_2D(cortex->nh_radius); k++) {
                        exc_count += ((current_neuron->synex_mask >> k) & 0x01u);
                        inh_count += (~(current_neuron->synex_mask >> k)) & 0x01u;
                    }
                    float exc_val = ((float) exc_count) / ((float) current_neuron->syn_count);
                    float inh_val = ((float) inh_count) / ((float) current_neuron->syn_count);

                    neuron_color = (Color) {
                        0xFFu * inh_val,
                        0xFFu * exc_val,
                        0xFFu * exc_val,
                        0xFFu * syn_count_value,
                    };
                    break;
                default:
                    break;
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