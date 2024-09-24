#include <behema/behema.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

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
                 bool drawInfo,
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

            if (drawInfo) {
                sf::Text pulseText;
                pulseText.setPosition(xNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.height + 6.0f);
                pulseText.setString(std::to_string(neuronValue));
                pulseText.setFont(font);
                pulseText.setCharacterSize(8);
                pulseText.setFillColor(sf::Color::White);
                if (neuronValue) {
                    window->draw(pulseText);
                }
            }

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

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 100;
    cortex_size_t cortex_height = 60;
    nh_radius_t nh_radius = 2;
    ticks_count_t sampleWindow = SAMPLE_WINDOW_MID;
    cv::Mat frame;
    cv::VideoCapture cam;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            cortex_width = atoi(argv[1]);
            break;
        case 3:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            break;
        case 4:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            break;
        case 5:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            sampleWindow = atoi(argv[4]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    cam.open(0);
    if (!cam.isOpened()) {
        printf("ERROR! Unable to open camera\n");
        return -1;
    }

    srand(time(NULL));

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create network model.
    cortex2d_t* even_cortex;
    cortex2d_t* odd_cortex;
    bhm_error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    error = c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }

    // Customize cortex properties.
    c2d_set_sample_window(even_cortex, sampleWindow);
    c2d_set_evol_step(even_cortex, 0x01U);
    c2d_set_pulse_mapping(even_cortex, PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(even_cortex, 24);
    c2d_copy(odd_cortex, even_cortex);


    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));

    initPositions(even_cortex, xNeuronPositions, yNeuronPositions);
    
    // create the window
    sf::RenderWindow window(desktopMode, "behema", sf::Style::Fullscreen);
    window.setMouseCursorVisible(false);
    
    bool feeding = false;
    bool showInfo = false;
    bool nDraw = true;
    bool sDraw = true;

    int counter = 0;

    // Inputs.
    input2d_t* leftEye;
    i2d_init(&leftEye, 0, 0, (cortex_width / 10) * 3, 1, DEFAULT_EXC_VALUE * 2, PULSE_MAPPING_FPROP);

    input2d_t* rightEye;
    i2d_init(&rightEye, (cortex_width / 10) * 7, 0, cortex_width, 1, DEFAULT_EXC_VALUE * 2, PULSE_MAPPING_FPROP);

    cv::Size eyeSize = cv::Size(leftEye->x1 - leftEye->x0, leftEye->y1 - leftEye->y0);

    // cortex_size_t lTimedInputsCoords[] = {0, cortex_height - 5, 1, cortex_height};
    // cortex_size_t rTimedInputsCoords[] = {cortex_width - 1, cortex_height - 5, cortex_width, cortex_height};

    char touchFileName[40];
    char inhexcFileName[40];
    sprintf(touchFileName, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    sprintf(inhexcFileName, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);

    c2d_touch_from_map(even_cortex, touchFileName);
    c2d_inhexc_from_map(even_cortex, inhexcFileName);

    ticks_count_t samplingBound = sampleWindow - 1;
    ticks_count_t sample_step = samplingBound;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    for (int i = 0; window.isOpen(); i++) {
        counter++;

        cortex2d_t* prev_cortex = i % 2 ? odd_cortex : even_cortex;
        cortex2d_t* next_cortex = i % 2 ? even_cortex : odd_cortex;

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
                        case sf::Keyboard::I:
                            showInfo = !showInfo;
                            break;
                        case sf::Keyboard::N:
                            nDraw = !nDraw;
                            break;
                        case sf::Keyboard::S:
                            sDraw = !sDraw;
                            break;
                        case sf::Keyboard::D:
                            c2d_to_file(prev_cortex, (char*) "out/test.c2d");
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
                    printf("ERROR! blank frame grabbed\n");
                    break;
                }

                cv::Mat resized;
                cv::resize(frame, resized, eyeSize);

                resized.at<uint8_t>(cv::Point(0, 0));
                for (cortex_size_t y = 0; y < eyeSize.height; y++) {
                    for (cortex_size_t x = 0; x < eyeSize.width; x++) {
                        cv::Vec3b val = resized.at<cv::Vec3b>(cv::Point(x, y));
                        leftEye->values[IDX2D(eyeSize.width - 1 - x, y, eyeSize.width)] = fmap(val[2],
                                                                                               0, 255,
                                                                                               0, samplingBound);
                        rightEye->values[IDX2D(x, y, eyeSize.width)] = fmap(val[0],
                                                                            0, 255,
                                                                            0, samplingBound);
                    }
                }

                // cv::resize(resized, frame, eyeSize * 15, 0, 0, cv::INTER_NEAREST);
                // cv::imshow("Preview", frame);
                // cv::waitKey(1);

                sample_step = 0;
            }

            // Feed the cortex.
            c2d_feed2d(prev_cortex, leftEye);
            c2d_feed2d(prev_cortex, rightEye);

            sample_step++;
        }

        // Clear the window with black color.
        window.clear(sf::Color(31, 31, 31, 255));

        // Draw synapses.
        if (sDraw) {
            drawSynapses(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions);
        }

        // Draw neurons.
        if (nDraw) {
            drawNeurons(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);
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
        c2d_tick(prev_cortex, next_cortex);
    }
    
    return 0;
}