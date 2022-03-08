#include <portia/portia.h>
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

            float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold);

            float radius = 2.0F + 5.0F * ((float) currentNeuron->pulse) / ((float) cortex->pulse_window);

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
                pulseText.setString(std::to_string(currentNeuron->pulse));
                pulseText.setFont(font);
                pulseText.setCharacterSize(8);
                pulseText.setFillColor(sf::Color::White);
                if (currentNeuron->pulse != 0) {
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
                        uint64_t syn_strength = (str_mask_a & 0x01U) |
                                                ((str_mask_b & 0x01U) << 0x01U) |
                                                ((str_mask_c & 0x01U) << 0x02U);

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (acMask & 1) {
                            sf::Vertex line[] = {
                                sf::Vertex(
                                    {xNeuronPositions[neighborIndex] * videoMode.width, yNeuronPositions[neighborIndex] * videoMode.height},
                                    excMask & 1
                                        ? sf::Color(31, 100, 127, 25 * (syn_strength + 1))
                                        : sf::Color(127, 100, 31, 25 * (syn_strength + 1))
                                ),
                                sf::Vertex(
                                    {xNeuronPositions[neuronIndex] * videoMode.width, yNeuronPositions[neuronIndex] * videoMode.height},
                                    excMask & 1
                                        ? sf::Color(31, 100, 127, 5 * (syn_strength + 1))
                                        : sf::Color(127, 100, 31, 5 * (syn_strength + 1))
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

void highlightNeuron(cortex2d_t* cortex,
                     sf::RenderWindow* window,
                     sf::VideoMode videoMode,
                     int* passedNeurons,
                     int* passedNeuronsSize,
                     float* xNeuronPositions,
                     float* yNeuronPositions,
                     int xFocus,
                     int yFocus) {
    passedNeurons[*passedNeuronsSize] = IDX2D(xFocus, yFocus, cortex->width);
    (*passedNeuronsSize)++;

    cortex_size_t nh_diameter = 2 * cortex->nh_radius + 1;

    int neuronIndex = IDX2D(xFocus, yFocus, cortex->width);

    neuron_t* currentNeuron = &(cortex->neurons[neuronIndex]);

    nh_mask_t acMask = currentNeuron->synac_mask;
    nh_mask_t excMask = currentNeuron->synex_mask;
    
    // Loop through neighbors.
    for (nh_radius_t y = 0; y < nh_diameter; y++) {
        for (nh_radius_t x = 0; x < nh_diameter; x++) {
            // Exclude the actual neuron from the list of neighbors.
            // Also exclude wrapping.
            if (!(x == cortex->nh_radius && y == cortex->nh_radius)) {
                // Fetch the current neighbor.
                cortex_size_t neighborIndex = IDX2D(WRAP(xFocus + (x - cortex->nh_radius), cortex->width),
                                                    WRAP(yFocus + (y - cortex->nh_radius), cortex->height),
                                                    cortex->width);

                bool passed = false;

                for (int i = 0; i < *passedNeuronsSize; i++) {
                    if (passedNeurons[i] == neighborIndex) {
                        passed = true;
                    }
                }

                if (acMask & 0x01U && !passed /*&& *passedNeuronsSize < 100*/) {
                    if ((xFocus + (x - cortex->nh_radius)) >= 0 &&
                        (xFocus + (x - cortex->nh_radius)) < cortex->width &&
                        (yFocus + (y - cortex->nh_radius)) >= 0 &&
                        (yFocus + (y - cortex->nh_radius)) < cortex->height) {
                        // Draw synapse.
                        sf::Vertex line[] = {
                            sf::Vertex(
                                {xNeuronPositions[neighborIndex] * videoMode.width, yNeuronPositions[neighborIndex] * videoMode.height},
                                sf::Color(255, 255, 255, 255)),
                            sf::Vertex(
                                {xNeuronPositions[neuronIndex] * videoMode.width, yNeuronPositions[neuronIndex] * videoMode.height},
                                sf::Color(255, 255, 255, 150))
                        };

                        window->draw(line, 2, sf::Lines);
                    }

                    // Recall.
                    highlightNeuron(cortex,
                                    window,
                                    videoMode,
                                    passedNeurons,
                                    passedNeuronsSize,
                                    xNeuronPositions,
                                    yNeuronPositions,
                                    WRAP(xFocus + (x - cortex->nh_radius), cortex->width),
                                    WRAP(yFocus + (y - cortex->nh_radius), cortex->height));
                    // return;
                }
            }

            // Shift the mask to check for the next neighbor.
            acMask >>= 1;
            excMask >>= 1;
        }
    }
}

int main(int argc, char **argv) {
    char* cortexFileName;
    // Input handling.
    switch (argc) {
        case 2:
            cortexFileName = argv[1];
            break;
        default:
            printf("USAGE: inspect <path/to/cortex/file>\n");
            exit(0);
            break;
    }

    // sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();
    sf::VideoMode desktopMode(800, 500);

    // Create network model.
    cortex2d_t cortex;
    c2d_from_file(&cortex, cortexFileName);

    float* xNeuronPositions = (float*) malloc(cortex.width * cortex.height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex.width * cortex.height * sizeof(float));

    initPositions(&cortex, xNeuronPositions, yNeuronPositions);
    
    // Create the window
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    sf::RenderWindow window(desktopMode, cortexFileName, sf::Style::Default, settings);
    
    bool showInfo = false;
    bool nDraw = true;
    bool sDraw = true;
    int xFocus = -1;
    int yFocus = -1;
    bool mouseIn = false;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    while (window.isOpen()) {
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
                case sf::Event::MouseEntered:
                    mouseIn = true;
                    break;
                case sf::Event::MouseLeft:
                    mouseIn = false;
                    break;
                case sf::Event::MouseMoved:
                    {
                        sf::Vector2i mousePos =  sf::Mouse::getPosition(window);
                        float xPos = ((float) mousePos.x) / ((float) window.getSize().x);
                        float yPos = ((float) mousePos.y) / ((float) window.getSize().y);
                        int xTmp = (int) (xPos * cortex.width);
                        int yTmp = (int) (yPos * cortex.height);
                        xFocus = xTmp;
                        yFocus = yTmp;
                    }
                    break;
                default:
                    break;
            }
        }

        // Clear the window with black color.
        window.clear(sf::Color(31, 31, 31, 255));

        // Draw synapses.
        if (sDraw) {
            drawSynapses(&cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions);
        }

        // Draw neurons.
        if (nDraw) {
            drawNeurons(&cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);
        }

        if (mouseIn && xFocus != -1 && yFocus != -1) {
            // Keep track of visited neurons.
            int passedNeurons[cortex.width * cortex.height];
            int* passedNeuronsSize = (int*) malloc(sizeof(int));
            (*passedNeuronsSize) = 0;

            highlightNeuron(&cortex,
                            &window,
                            desktopMode,
                            passedNeurons,
                            passedNeuronsSize,
                            xNeuronPositions,
                            yNeuronPositions,
                            xFocus,
                            yFocus);
        }

        // End the current frame.
        window.display();
    }


    return 0;
}