#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <portia/portia.h>

void initPositions(cortex2d_t* cortex, float* xNeuronPositions, float* yNeuronPositions) {
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            xNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) x) + 0.5f) / (float) cortex->width;
            yNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) y) + 0.5f) / (float) cortex->height;
        }
    }
}

void drawNeurons(cortex2d_t* cortex,
                 sf::RenderWindow* window,
                 sf::VideoMode videoMode,
                 float* xNeuronPositions,
                 float* yNeuronPositions,
                 sf::VideoMode desktopMode,
                 sf::Font font) {
    for (cortex_size_t i = 0; i < cortex->height; i++) {
        for (cortex_size_t j = 0; j < cortex->width; j++) {
            sf::CircleShape neuronSpot;

            neuron_t* currentNeuron = &(cortex->neurons[IDX2D(j, i, cortex->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold + (float) (currentNeuron->pulse));
            // float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold);

            bool fired = currentNeuron->pulse_mask & 0x01U;

            float radius = 1.0F + 6.0F * ((float) currentNeuron->pulse) / ((float) cortex->pulse_window);

            neuronSpot.setRadius(radius);

            if (fired) {
                neuronSpot.setFillColor(sf::Color::White);
            } else {
                if (neuronValue < 0) {
                    neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 - 31 * neuronValue));
                } else {
                    neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 + 224 * neuronValue));
                }
            }
            
            neuronSpot.setPosition(xNeuronPositions[IDX2D(j, i, cortex->width)] * videoMode.width, yNeuronPositions[IDX2D(j, i, cortex->width)] * videoMode.height);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);

            window->draw(neuronSpot);
        }
    }
}

void drawSynapses(cortex2d_t* cortex, sf::RenderWindow* window, sf::VideoMode videoMode, float* xNeuronPositions, float* yNeuronPositions) {
    for (cortex_size_t i = 0; i < cortex->height; i++) {
        for (cortex_size_t j = 0; j < cortex->width; j++) {
            cortex_size_t neuronIndex = IDX2D(j, i, cortex->width);
            neuron_t* currentNeuron = &(cortex->neurons[neuronIndex]);

            cortex_size_t nh_diameter = 2 * cortex->nh_radius + 1;

            nh_mask_t acMask = currentNeuron->synac_mask;
            nh_mask_t excMask = currentNeuron->synex_mask;
            nh_mask_t str_mask_a = currentNeuron->synstr_mask_a;
            nh_mask_t str_mask_b = currentNeuron->synstr_mask_b;
            nh_mask_t str_mask_c = currentNeuron->synstr_mask_c;
            
            for (nh_radius_t k = 0; k < nh_diameter; k++) {
                for (nh_radius_t l = 0; l < nh_diameter; l++) {
                    // Exclude the actual neuron from the list of neighbors.
                    // Also exclude wrapping.
                    if (!(k == cortex->nh_radius && l == cortex->nh_radius) &&
                        (j + (l - cortex->nh_radius)) >= 0 &&
                        (j + (l - cortex->nh_radius)) < cortex->width &&
                        (i + (k - cortex->nh_radius)) >= 0 &&
                        (i + (k - cortex->nh_radius)) < cortex->height) {
                        // Fetch the current neighbor.
                        cortex_size_t neighborIndex = IDX2D(WRAP(j + (l - cortex->nh_radius), cortex->width),
                                                            WRAP(i + (k - cortex->nh_radius), cortex->height),
                                                            cortex->width);

                        // Compute the current synapse strength.
                        syn_strength_t syn_strength = (str_mask_a & 0x01U) |
                                                ((str_mask_b & 0x01U) << 0x01U) |
                                                ((str_mask_c & 0x01U) << 0x02U);

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (acMask & 1) {
                            sf::Vertex line[] = {
                                sf::Vertex(
                                    {xNeuronPositions[neighborIndex] * videoMode.width, yNeuronPositions[neighborIndex] * videoMode.height},
                                    excMask & 1
                                        ? sf::Color(31, 100, 127, 5 * (syn_strength + 1))
                                        : sf::Color(127, 100, 31, 5 * (syn_strength + 1))
                                ),
                                sf::Vertex(
                                    {xNeuronPositions[neuronIndex] * videoMode.width, yNeuronPositions[neuronIndex] * videoMode.height},
                                    excMask & 1
                                        ? sf::Color(31, 100, 127, 25 * (syn_strength + 1))
                                        : sf::Color(127, 100, 31, 25 * (syn_strength + 1))
                                )
                            };

                            window->draw(line, 2, sf::Lines);
                        }
                    }

                    // Shift the mask to check for the next neighbor.
                    acMask >>= 1;
                    excMask >>= 1;
                    str_mask_a >>= 1;
                    str_mask_b >>= 1;
                    str_mask_c >>= 1;
                }
            }
        }
    }
}



void setup_cortexes(cortex2d_t** even_cortex,
                    cortex2d_t** odd_cortex,
                    cortex_size_t cortex_width,
                    cortex_size_t cortex_height,
                    nh_radius_t nh_radius) {
    c2d_init(even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_init(odd_cortex, cortex_width, cortex_height, nh_radius);

    // Customize cortex properties.
    // c2d_set_sample_window(*even_cortex, sampleWindow);
    c2d_set_evol_step(*even_cortex, 0x01U);
    c2d_set_pulse_mapping(*even_cortex, PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(*even_cortex, 24);
    char touchFileName[40];
    char inhexcFileName[40];
    sprintf(touchFileName, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    sprintf(inhexcFileName, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    c2d_touch_from_map(*even_cortex, touchFileName);
    c2d_inhexc_from_map(*even_cortex, inhexcFileName);
    c2d_copy(*odd_cortex, *even_cortex);

}

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 100;
    cortex_size_t cortex_height = 60;
    cortex_size_t input_width = 32;
    cortex_size_t input_height = 1;
    dim3 cortex_grid_size(cortex_width / BLOCK_SIZE_2D, cortex_height / BLOCK_SIZE_2D, 1);
    dim3 cortex_block_size(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
    dim3 input_grid_size(input_width, input_height, 1);
    dim3 input_block_size(1, 1, 1);
    nh_radius_t nh_radius = 2;
    ticks_count_t sampleWindow = SAMPLE_WINDOW_MID;
    cv::Mat frame;
    cv::VideoCapture cam;

    cam.open(0);
    if (!cam.isOpened()) {
        printf("ERROR! Unable to open camera\n");
        return -1;
    }

    srand(time(NULL));

    error_code_t error;

    // Cortex configuration.
    cortex2d_t* even_cortex;
    cortex2d_t* odd_cortex;
    setup_cortexes(
        &even_cortex,
        &odd_cortex,
        cortex_width,
        cortex_height,
        nh_radius
    );

    // Copy cortexes to device.
    cortex2d_t* d_odd_cortex;
    cortex2d_t* d_even_cortex;
    cudaMalloc((void**) &d_even_cortex, sizeof(cortex2d_t));
    cudaCheckError();
    cudaMalloc((void**) &d_odd_cortex, sizeof(cortex2d_t));
    cudaCheckError();
    error = c2d_to_device(d_even_cortex, even_cortex);
    error = c2d_to_device(d_odd_cortex, odd_cortex);

    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    initPositions(even_cortex, xNeuronPositions, yNeuronPositions);
    
    // Create a new window.
    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();
    sf::RenderWindow window(desktopMode, "Portia", sf::Style::Fullscreen);
    window.setMouseCursorVisible(false);
    
    bool feeding = false;
    bool nDraw = true;
    bool sDraw = true;

    int counter = 0;

    // Inputs.
    input2d_t* left_eye;
    i2d_init(
        &left_eye,
        0,
        0,
        input_width,
        input_height,
        DEFAULT_EXC_VALUE * 2,
        PULSE_MAPPING_FPROP
    );

    input2d_t* right_eye;
    i2d_init(&right_eye,
             cortex_width - input_width,
             0,
             cortex_width,
             input_height,
             DEFAULT_EXC_VALUE * 2,
             PULSE_MAPPING_FPROP);

    // Set input values.
    for (int i = 0; i < input_width * input_height; i++) {
        left_eye->values[i] = even_cortex->sample_window - 1;
        right_eye->values[i] = even_cortex->sample_window - 1;
    }

    // Copy input to device.
    input2d_t* d_left_eye;
    input2d_t* d_right_eye;
    cudaMalloc((void**) &d_left_eye, sizeof(input2d_t));
    cudaCheckError();
    i2d_to_device(d_left_eye, left_eye);
    cudaMalloc((void**) &d_right_eye, sizeof(input2d_t));
    cudaCheckError();
    i2d_to_device(d_right_eye, right_eye);

    cv::Size eyeSize = cv::Size(input_width, input_height);

    ticks_count_t samplingBound = sampleWindow - 1;
    ticks_count_t sample_step = samplingBound;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    cortex2d_t* h_cortex;
    error = c2d_init(&h_cortex, cortex_width, cortex_height, nh_radius);
    c2d_copy(h_cortex, even_cortex);

    for (int i = 0; window.isOpen(); i++) {
        counter++;

        cortex2d_t* prev_cortex = i % 2 ? d_odd_cortex : d_even_cortex;
        cortex2d_t* next_cortex = i % 2 ? d_even_cortex : d_odd_cortex;

        // Check all the window's events that were triggered since the last iteration of the loop.
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
                case sf::Event::Closed:
                    // Close requested event: close the window.
                    window.close();
                    break;
                case sf::Event::KeyReleased:
                    switch (event.key.code) {
                        case sf::Keyboard::Escape:
                        case sf::Keyboard::Q:
                            window.close();
                            break;
                        case sf::Keyboard::Space:
                            feeding = !feeding;
                            break;
                        case sf::Keyboard::N:
                            nDraw = !nDraw;
                            break;
                        case sf::Keyboard::S:
                            sDraw = !sDraw;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        // Only get new inputs according to the sample rate.
        if (feeding) {
            if (sample_step > samplingBound) {
                // Fetch input.
                cam.read(frame);

                if (frame.empty()) {
                    printf("\nERROR! Blank frame grabbed\n");
                    break;
                }

                cv::Mat resized;
                cv::resize(frame, resized, eyeSize);

                resized.at<uint8_t>(cv::Point(0, 0));
                for (cortex_size_t y = 0; y < input_height; y++) {
                    for (cortex_size_t x = 0; x < input_width; x++) {
                        cv::Vec3b val = resized.at<cv::Vec3b>(cv::Point(x, y));
                        left_eye->values[IDX2D(input_width - 1 - x, y, input_width)] = fmap(val[2],
                                                                                            0, 255,
                                                                                            0, samplingBound);
                        right_eye->values[IDX2D(x, y, input_width)] = fmap(val[0],
                                                                           0, 255,
                                                                           0, samplingBound);
                    }
                }

                // Copy input to device.
                i2d_to_device(d_left_eye, left_eye);
                cudaCheckError();   
                i2d_to_device(d_right_eye, right_eye);
                cudaCheckError();

                sample_step = 0;
            }

            // Feed the cortex.
            c2d_feed2d<<<input_grid_size, input_block_size>>>(prev_cortex, d_left_eye);
            cudaCheckError();
            cudaDeviceSynchronize();
            c2d_feed2d<<<input_grid_size, input_block_size>>>(prev_cortex, d_right_eye);
            cudaCheckError();
            cudaDeviceSynchronize();

            // printf("\n4\n");

            sample_step++;
        }

        // Clear the window with black color.
        window.clear(sf::Color(31, 31, 31, 255));

        // Copy cortex back to host in order to print it.
        error = c2d_to_host(h_cortex, next_cortex);

        // Draw synapses.
        if (sDraw) {
            drawSynapses(h_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions);
        }

        // Draw neurons.
        if (nDraw) {
            drawNeurons(h_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions, desktopMode, font);
        }

        sf::Text text;
        text.setPosition(10.0, 10.0);
        text.setFont(font);
        char string[100];
        snprintf(string, 100, "%d", sample_step);
        text.setString(string);
        text.setCharacterSize(12);
        text.setFillColor(sf::Color::White);
        window.draw(text);

        // End the current frame.
        window.display();

        // usleep(10000);

        // Tick the cortex.
        c2d_tick<<<cortex_grid_size, cortex_block_size>>>(prev_cortex, next_cortex);
        cudaCheckError();
        cudaDeviceSynchronize();
    }

    // Cleanup.
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    c2d_device_destroy(d_even_cortex);
    c2d_device_destroy(d_odd_cortex);
    i2d_destroy(left_eye);
    i2d_destroy(right_eye);
    i2d_device_destroy(d_left_eye);
    i2d_device_destroy(d_right_eye);
    
    return 0;
}