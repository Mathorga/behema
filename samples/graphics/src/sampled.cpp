#include <portia/portia.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

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

            nh_mask_t nb_mask = currentNeuron->synac_mask;
            
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

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (nb_mask & 0x01) {
                            sf::Vertex line[] = {
                                sf::Vertex(
                                    {xNeuronPositions[neighborIndex] * videoMode.width, yNeuronPositions[neighborIndex] * videoMode.height},
                                    sf::Color(255, 127, 31, 10)),
                                sf::Vertex(
                                    {xNeuronPositions[neuronIndex] * videoMode.width, yNeuronPositions[neuronIndex] * videoMode.height},
                                    sf::Color(31, 127, 255, 50))
                            };

                            window->draw(line, 2, sf::Lines);
                        }
                    }

                    // Shift the mask to check for the next neighbor.
                    nb_mask >>= 1;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 100;
    cortex_size_t cortex_height = 60;
    nh_radius_t nh_radius = 2;

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
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    srand(time(NULL));

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_set_evol_step(&even_cortex, 0x0A);
    c2d_set_pulse_window(&even_cortex, 0x3A);
    c2d_set_syngen_beat(&even_cortex, 0.1F);
    c2d_set_max_touch(&even_cortex, 0.2F);
    c2d_set_sample_window(&even_cortex, 100);
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
    int renderingInterval = 1;

    // Coordinates for input neurons.
    cortex_size_t lInputsCoords[] = {10, 5, 40, 20};
    cortex_size_t rInputsCoords[] = {even_cortex.width - 40 , 5, even_cortex.width - 10, 20};

    ticks_count_t* lInputs = (ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t* rInputs = (ticks_count_t*) malloc((rInputsCoords[2] - rInputsCoords[0]) * (rInputsCoords[3] - rInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t sample_step = even_cortex.sample_window - 1;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    // Run the program as long as the window is open.
    for (int i = 0; window.isOpen(); i++) {
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
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        counter++;

        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        // Only get new inputs according to the sample rate.
        if (feeding) {
            if (sample_step > prev_cortex->sample_window - 1) {
                // Fetch input.
                for (cortex_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                    for (cortex_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                        lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])] = (rand() % (prev_cortex->sample_window - 1));
                    }
                }
                for (cortex_size_t y = rInputsCoords[1]; y < rInputsCoords[3]; y++) {
                    for (cortex_size_t x = rInputsCoords[0]; x < rInputsCoords[2]; x++) {
                        rInputs[IDX2D(x - rInputsCoords[0], y - rInputsCoords[1], rInputsCoords[2] - rInputsCoords[0])] = (rand() % (prev_cortex->sample_window - 1));
                    }
                }
                sample_step = 0;
            }

            // Feed the cortex.
            c2d_sample_sqfeed(prev_cortex, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, DEFAULT_EXC_VALUE);
            c2d_sample_sqfeed(prev_cortex, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, DEFAULT_EXC_VALUE);

            sample_step++;
        }

        if (counter % renderingInterval == 0) {
            // Clear the window with black color.
            window.clear(sf::Color(31, 31, 31, 255));

            // Highlight input neurons.
            for (cortex_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (cortex_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    sf::CircleShape neuronCircle;

                    float radius = 6.0f;
                    neuronCircle.setRadius(radius);
                    neuronCircle.setOutlineThickness(1);
                    neuronCircle.setOutlineColor(sf::Color(255, 255, 255, 32));

                    neuronCircle.setFillColor(sf::Color::Transparent);
                    
                    neuronCircle.setPosition(xNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.width, yNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.height);

                    // Center the spot.
                    neuronCircle.setOrigin(radius, radius);
                    window.draw(neuronCircle);

                    // sf::Text pulseText;
                    // pulseText.setPosition(xNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.height + 6.0f);
                    // pulseText.setString(std::to_string(lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])]));
                    // pulseText.setFont(font);
                    // pulseText.setCharacterSize(8);
                    // pulseText.setFillColor(sf::Color::White);
                    // window.draw(pulseText);
                }
            }

            // Highlight input neurons.
            for (cortex_size_t y = rInputsCoords[1]; y < rInputsCoords[3]; y++) {
                for (cortex_size_t x = rInputsCoords[0]; x < rInputsCoords[2]; x++) {
                    sf::CircleShape neuronCircle;

                    float radius = 6.0f;
                    neuronCircle.setRadius(radius);
                    neuronCircle.setOutlineThickness(1);
                    neuronCircle.setOutlineColor(sf::Color(255, 255, 255, 32));

                    neuronCircle.setFillColor(sf::Color::Transparent);
                    
                    neuronCircle.setPosition(xNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.width, yNeuronPositions[IDX2D(x, y, prev_cortex->width)] * desktopMode.height);

                    // Center the spot.
                    neuronCircle.setOrigin(radius, radius);
                    window.draw(neuronCircle);
                }
            }

            // Draw neurons.
            if (nDraw) {
                drawNeurons(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);
            }

            // Draw synapses.
            if (sDraw) {
                drawSynapses(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions);
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
            
            usleep(5000);
        }

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);
    }

    return 0;
}
