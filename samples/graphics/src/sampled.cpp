#include <hal/hal.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

void initPositions(field2d_t* field, float* xNeuronPositions, float* yNeuronPositions) {
    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            xNeuronPositions[IDX2D(x, y, field->width)] = (((float) x) + 0.5f) / (float) field->width;
            yNeuronPositions[IDX2D(x, y, field->width)] = (((float) y) + 0.5f) / (float) field->height;
        }
    }
}

void drawNeurons(field2d_t* field,
                 sf::RenderWindow* window,
                 sf::VideoMode videoMode,
                 float* xNeuronPositions,
                 float* yNeuronPositions,
                 bool drawInfo,
                 sf::VideoMode desktopMode,
                 sf::Font font) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            sf::CircleShape neuronSpot;

            neuron_t* currentNeuron = &(field->neurons[IDX2D(j, i, field->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) field->fire_threshold);

            float radius = 3.0f;

            neuronSpot.setRadius(radius);

            if (neuronValue < 0) {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 - 31 * neuronValue));
            } else if (currentNeuron->value > field->fire_threshold) {
                neuronSpot.setFillColor(sf::Color::White);
            } else {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 + 224 * neuronValue));
            }
            
            neuronSpot.setPosition(xNeuronPositions[IDX2D(j, i, field->width)] * videoMode.width, yNeuronPositions[IDX2D(j, i, field->width)] * videoMode.height);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);

            if (drawInfo) {
                sf::Text pulseText;
                pulseText.setPosition(xNeuronPositions[IDX2D(j, i, field->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(j, i, field->width)] * desktopMode.height + 6.0f);
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

void drawSynapses(field2d_t* field, sf::RenderWindow* window, sf::VideoMode videoMode, float* xNeuronPositions, float* yNeuronPositions) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            field_size_t neuronIndex = IDX2D(j, i, field->width);
            neuron_t* currentNeuron = &(field->neurons[neuronIndex]);

            field_size_t nh_diameter = 2 * field->nh_radius + 1;

            nh_mask_t nb_mask = currentNeuron->nh_mask;
            
            for (nh_radius_t k = 0; k < nh_diameter; k++) {
                for (nh_radius_t l = 0; l < nh_diameter; l++) {
                    // Exclude the actual neuron from the list of neighbors.
                    // Also exclude wrapping.
                    if (!(k == field->nh_radius && l == field->nh_radius) &&
                        (j + (l - field->nh_radius)) >= 0 &&
                        (j + (l - field->nh_radius)) < field->width &&
                        (i + (k - field->nh_radius)) >= 0 &&
                        (i + (k - field->nh_radius)) < field->height) {
                        // Fetch the current neighbor.
                        field_size_t neighborIndex = IDX2D(WRAP(j + (l - field->nh_radius), field->width),
                                                           WRAP(i + (k - field->nh_radius), field->height),
                                                           field->width);

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
    field_size_t field_width = 100;
    field_size_t field_height = 60;
    nh_radius_t nh_radius = 2;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            field_width = atoi(argv[1]);
            break;
        case 3:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            break;
        case 4:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
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
    field2d_t even_field;
    field2d_t odd_field;
    f2d_init(&even_field, field_width, field_height, nh_radius);
    f2d_set_evol_step(&even_field, 0x20u);
    f2d_set_pulse_window(&even_field, 0x3A);
    // f2d_set_syngen_beat(&even_field, 0.3F);
    f2d_set_max_touch(&even_field, 0.2F);
    f2d_set_sample_window(&even_field, 10);
    odd_field = *f2d_copy(&even_field);

    float* xNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));

    initPositions(&even_field, xNeuronPositions, yNeuronPositions);
    
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
    field_size_t lInputsCoords[] = {5, 5, 30, 20};
    field_size_t rInputsCoords[] = {50, 5, 75, 20};

    ticks_count_t* lInputs = (ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t* rInputs = (ticks_count_t*) malloc((rInputsCoords[2] - rInputsCoords[0]) * (rInputsCoords[3] - rInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t sample_step = 0;

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

        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        // Only get new inputs according to the sample rate.
        if (sample_step == prev_field->sample_window) {
            // Fetch input.
            for (field_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (field_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])] = (rand() % (prev_field->sample_window - 1));
                }
            }
            for (field_size_t y = rInputsCoords[1]; y < rInputsCoords[3]; y++) {
                for (field_size_t x = rInputsCoords[0]; x < rInputsCoords[2]; x++) {
                    rInputs[IDX2D(x - rInputsCoords[0], y - rInputsCoords[1], rInputsCoords[2] - rInputsCoords[0])] = (rand() % (prev_field->sample_window - 1));
                }
            }
            sample_step = 0;
        }

        // Feed the field.
        if (feeding) {
            f2d_sample_sqfeed(prev_field, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, DEFAULT_CHARGE_VALUE);
            f2d_sample_sqfeed(prev_field, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, DEFAULT_CHARGE_VALUE);
        }

        if (counter % renderingInterval == 0) {
            // Clear the window with black color.
            window.clear(sf::Color(31, 31, 31, 255));

            // Highlight input neurons.
            for (field_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (field_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    sf::CircleShape neuronCircle;

                    float radius = 6.0f;
                    neuronCircle.setRadius(radius);
                    neuronCircle.setOutlineThickness(1);
                    neuronCircle.setOutlineColor(sf::Color(255, 255, 255, 64));

                    neuronCircle.setFillColor(sf::Color::Transparent);
                    
                    neuronCircle.setPosition(xNeuronPositions[IDX2D(x, y, prev_field->width)] * desktopMode.width, yNeuronPositions[IDX2D(x, y, prev_field->width)] * desktopMode.height);

                    // Center the spot.
                    neuronCircle.setOrigin(radius, radius);
                    window.draw(neuronCircle);
                }
            }

            // Draw neurons.
            if (nDraw) {
                drawNeurons(next_field, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);
            }

            // Draw synapses.
            if (sDraw) {
                drawSynapses(next_field, &window, desktopMode, xNeuronPositions, yNeuronPositions);
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

        // Tick the field.
        f2d_tick(prev_field, next_field);

        sample_step++;
    }

    return 0;
}