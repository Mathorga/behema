#include <portia/portia.h>
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

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    double slope = ((double) output_end - (double) output_start) / ((double) input_end - (double) input_start);
    return (double) output_start + slope * ((double) input - (double) input_start);
}

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

            float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold);

            float radius = 2.0F + 5.0F * ((float) currentNeuron->tick_pulse) / ((float) cortex->pulse_window);

            neuronSpot.setRadius(radius);

            if (neuronValue < 0) {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 - 31 * neuronValue));
            } else if (currentNeuron->value > cortex->fire_threshold) {
                neuronSpot.setFillColor(sf::Color::White);
            } else {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 + 224 * neuronValue));
            }
            
            neuronSpot.setPosition(xNeuronPositions[IDX2D(j, i, cortex->width)] * videoMode.width, yNeuronPositions[IDX2D(j, i, cortex->width)] * videoMode.height);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);

            if (drawInfo) {
                sf::Text pulseText;
                pulseText.setPosition(xNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.height + 6.0f);
                pulseText.setString(std::to_string(currentNeuron->tick_pulse));
                pulseText.setFont(font);
                pulseText.setCharacterSize(8);
                pulseText.setFillColor(sf::Color::White);
                if (currentNeuron->tick_pulse != 0) {
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
    ticks_count_t sampleWindow = 10;
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

    ticks_count_t samplingBound = sampleWindow - 1;

    cam.open(0);
    if (!cam.isOpened()) {
        printf("ERROR! Unable to open camera\n");
        return -1;
    }

    srand(time(NULL));

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    c2d_set_sample_window(&even_cortex, sampleWindow);
    c2d_set_evol_step(&even_cortex, 0x01U);
    c2d_set_pulse_mapping(&even_cortex, PULSE_MAPPING_FPROP);
    c2d_set_max_syn_count(&even_cortex, 24);
    c2d_copy(&odd_cortex, &even_cortex);

    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));

    initPositions(&even_cortex, xNeuronPositions, yNeuronPositions);
    
    // create the window
    sf::RenderWindow window(desktopMode, "Liath", sf::Style::Fullscreen);
    window.setMouseCursorVisible(false);
    
    bool feeding = false;
    bool showInfo = false;
    bool nDraw = true;
    bool sDraw = true;

    int counter = 0;

    // Coordinates for input neurons.
    cortex_size_t rInputsCoords[] = {0, 0, (cortex_width / 10) * 3, 1};
    ticks_count_t* rInputs = (ticks_count_t*) malloc((rInputsCoords[2] - rInputsCoords[0]) * (rInputsCoords[3] - rInputsCoords[1]) * sizeof(ticks_count_t));
    cv::Size rInputSize = cv::Size(rInputsCoords[2] - rInputsCoords[0], rInputsCoords[3] - rInputsCoords[1]);

    cortex_size_t bInputsCoords[] = {(cortex_width / 10) * 7, 0, cortex_width, 1};
    ticks_count_t* bInputs = (ticks_count_t*) malloc((bInputsCoords[2] - bInputsCoords[0]) * (bInputsCoords[3] - bInputsCoords[1]) * sizeof(ticks_count_t));
    cv::Size bInputSize = cv::Size(bInputsCoords[2] - bInputsCoords[0], bInputsCoords[3] - bInputsCoords[1]);

    cortex_size_t lTimedInputsCoords[] = {0, cortex_height - 5, 1, cortex_height};
    cortex_size_t rTimedInputsCoords[] = {cortex_width - 1, cortex_height - 5, cortex_width, cortex_height};

    c2d_touch_from_map(&even_cortex, "./res/100_60_touch.pgm");
    c2d_inhexc_from_map(&even_cortex, "./res/100_60_inhexc.pgm");

    ticks_count_t sample_step = samplingBound;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    for (int i = 0; window.isOpen(); i++) {
        counter++;

        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

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
                cv::resize(frame, resized, rInputSize);
                
                // cv::cvtColor(resized, frame, cv::COLOR_BGR2GRAY);

                resized.at<uint8_t>(cv::Point(0, 0));
                for (cortex_size_t y = 0; y < rInputSize.height; y++) {
                    for (cortex_size_t x = 0; x < rInputSize.width; x++) {
                        cv::Vec3b val = resized.at<cv::Vec3b>(cv::Point(x, y));
                        rInputs[IDX2D(x, y, rInputSize.width)] = map(val[2],
                                                                     0, 255,
                                                                     0, even_cortex.sample_window - 1);
                        bInputs[IDX2D(x, y, rInputSize.width)] = map(val[0],
                                                                     0, 255,
                                                                     0, even_cortex.sample_window - 1);
                    }
                }

                sample_step = 0;

                // cv::resize(resized, frame, rInputSize * 15, 0, 0, cv::INTER_NEAREST);
                // cv::imshow("Preview", frame);
                // cv::waitKey(1);
            }

            // Feed the cortex.
            c2d_sample_sqfeed(prev_cortex, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, DEFAULT_EXCITING_VALUE * 4);
            c2d_sample_sqfeed(prev_cortex, bInputsCoords[0], bInputsCoords[1], bInputsCoords[2], bInputsCoords[3], sample_step, bInputs, DEFAULT_EXCITING_VALUE * 4);

            c2d_sqfeed(prev_cortex, lTimedInputsCoords[0], lTimedInputsCoords[1], lTimedInputsCoords[2], lTimedInputsCoords[3], 0x14);
            c2d_sqfeed(prev_cortex, rTimedInputsCoords[0], rTimedInputsCoords[1], rTimedInputsCoords[2], rTimedInputsCoords[3], 0x14);

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

        // usleep(5000);

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);
    }
    
    return 0;
}