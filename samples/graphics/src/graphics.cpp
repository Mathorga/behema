#include <portia/portia.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    float range = max - min;
    return (random * range) + min;
}

void initPositions(cortex2d_t* cortex, float* xNeuronPositions, float* yNeuronPositions, bool random) {
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            if (random) {
                xNeuronPositions[IDX2D(x, y, cortex->width)] = randomFloat(0, 1);
                yNeuronPositions[IDX2D(x, y, cortex->width)] = randomFloat(0, 1);
            } else {
                xNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) x) + 0.5f) / (float) cortex->width;
                yNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) y) + 0.5f) / (float) cortex->height;
            }
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

            float radius = 3.0f;

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
                sf::Text valueText;
                valueText.setPosition(xNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(j, i, cortex->width)] * desktopMode.height + 6.0f);
                valueText.setString(std::to_string(currentNeuron->value));
                valueText.setFont(font);
                valueText.setCharacterSize(8);
                valueText.setFillColor(sf::Color::White);
                window->draw(valueText);
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
                    if (!(k == cortex->nh_radius && l == cortex->nh_radius)) {
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
                                    sf::Color(31, 127, 255, 10))
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
    cortex_size_t inputs_count = 30;
    cortex_size_t inputs_spread = 4;

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
            inputs_count = atoi(argv[4]);
            break;
        default:
            printf("USAGE: graphics <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }
    
    bool feeding = false;
    bool showInfo = false;
    bool randomPositions = true;

    int counter = 0;
    int renderingInterval = 1;

    srand(time(0));

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_copy(&odd_cortex, &even_cortex);

    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));

    initPositions(&even_cortex, xNeuronPositions, yNeuronPositions, randomPositions);
    
    sf::ContextSettings settings;
    // settings.antialiasingLevel = 16;

    // create the window
    sf::RenderWindow window(desktopMode, "Liath", sf::Style::Fullscreen, settings);

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    // Run the program as long as the window is open.
    for (int i = 0; window.isOpen(); i++) {
        usleep(5000);
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
                        case sf::Keyboard::R:
                            randomPositions = !randomPositions;
                            initPositions(prev_cortex, xNeuronPositions, yNeuronPositions, randomPositions);
                            break;
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
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        // Feed the cortex.
        if (feeding && rand() % 100 > 10) {
            c2d_rsfeed(prev_cortex, 0, inputs_count, 2 * DEFAULT_EXCITING_VALUE, inputs_spread);
        }

        if (counter % renderingInterval == 0) {
            // Clear the window with black color.
            window.clear(sf::Color(31, 31, 31, 255));

            // Highlight input neurons.
            for (cortex_size_t i = 0; i < inputs_count; i++) {
                sf::CircleShape neuronCircle;

                float radius = 10.0f;
                neuronCircle.setRadius(radius);
                neuronCircle.setOutlineThickness(2);
                neuronCircle.setOutlineColor(sf::Color::White);

                neuronCircle.setFillColor(sf::Color::Transparent);
                
                neuronCircle.setPosition(xNeuronPositions[i * inputs_spread] * desktopMode.width, yNeuronPositions[i * inputs_spread] * desktopMode.height);

                // Center the spot.
                neuronCircle.setOrigin(radius, radius);
                window.draw(neuronCircle);
            }

            // Draw neurons.
            drawNeurons(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);

            // Draw synapses.
            drawSynapses(next_cortex, &window, desktopMode, xNeuronPositions, yNeuronPositions);

            // End the current frame.
            window.display();
        }

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);
    }
    return 0;
}
