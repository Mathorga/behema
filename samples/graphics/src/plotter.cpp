#include <liath/liath.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    float range = max - min;
    return (random * range) + min;
}

void initPositions(field2d_t* field, float* xNeuronPositions, float* yNeuronPositions, bool random) {
    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            if (random) {
                xNeuronPositions[IDX2D(x, y, field->width)] = randomFloat(0, 1);
                yNeuronPositions[IDX2D(x, y, field->width)] = randomFloat(0, 1);
            } else {
                xNeuronPositions[IDX2D(x, y, field->width)] = (((float) x) + 0.5f) / (float) field->width;
                yNeuronPositions[IDX2D(x, y, field->width)] = (((float) y) + 0.5f) / (float) field->height;
            }
        }
    }
}

void drawNeurons(field2d_t* field,
                 sf::RenderTexture* texture,
                 uint32_t textureWidth,
                 uint32_t textureHeight,
                 float* xNeuronPositions,
                 float* yNeuronPositions,
                 sf::Font font) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            sf::CircleShape neuronSpot;

            neuron_t* currentNeuron = &(field->neurons[IDX2D(j, i, field->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) field->fire_threshold);
            float neuronInfluence = ((float) currentNeuron->influence) / ((float) NEURON_MAX_INFLUENCE);

            float radius = 2.0f + neuronInfluence * 3.0f;

            neuronSpot.setRadius(radius);

            if (neuronValue < 0) {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 - 31 * neuronValue));
            } else if (currentNeuron->value > field->fire_threshold) {
                neuronSpot.setFillColor(sf::Color::White);
            } else {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 + 224 * neuronValue));
            }
            
            neuronSpot.setPosition(xNeuronPositions[IDX2D(j, i, field->width)] * textureWidth, yNeuronPositions[IDX2D(j, i, field->width)] * textureHeight);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);

            texture->draw(neuronSpot);
        }
    }
}

void drawSynapses(field2d_t* field, sf::RenderTexture* texture, uint32_t textureWidth, uint32_t textureHeight, float* xNeuronPositions, float* yNeuronPositions) {
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
                                    {xNeuronPositions[neighborIndex] * textureWidth, yNeuronPositions[neighborIndex] * textureHeight},
                                    sf::Color(255, 127, 31, 10)),
                                sf::Vertex(
                                    {xNeuronPositions[neuronIndex] * textureWidth, yNeuronPositions[neuronIndex] * textureHeight},
                                    sf::Color(31, 127, 255, 50))
                            };

                            texture->draw(line, 2, sf::Lines);
                        }
                    }

                    // Shift the mask to check for the next neighbor.
                    nb_mask >>= 1;
                }
            }
        }
    }
}

void plot(sf::RenderTexture* texture) {
    char fileName[100];
    snprintf(fileName, 100, "out/%lu.bmp", (unsigned long) time(NULL));
    texture->getTexture().copyToImage().saveToFile(fileName);
}

int main(int argc, char **argv) {
    field_size_t field_width = 100;
    field_size_t field_height = 60;
    nh_radius_t nh_radius = 2;
    field_size_t inputs_count = 131;
    ticks_count_t plotInterval = 5000;

    uint32_t textureWidth = 1366;
    uint32_t textureHeight = 768;

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
        case 5:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            inputs_count = atoi(argv[4]);
            break;
        case 6:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            inputs_count = atoi(argv[4]);
            plotInterval = atoi(argv[5]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    srand(time(0));

    // Create network model.
    field2d_t even_field;
    field2d_t odd_field;
    f2d_init(&even_field, field_width, field_height, nh_radius);
    odd_field = *f2d_copy(&even_field);

    float* xNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));

    initPositions(&even_field, xNeuronPositions, yNeuronPositions, false);
    
    // Create the texture to render.
    sf::RenderTexture* tex = new sf::RenderTexture();
    tex->create(textureWidth, textureHeight);
    
    bool feeding = true;

    int counter = 0;

    ticks_count_t sample_rate = 10;

    ticks_count_t* inputs = (ticks_count_t*) malloc(inputs_count * sizeof(ticks_count_t));
    ticks_count_t samples_count = 0;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    bool running = true;

    // Run the program as long as the window is open.
    for (int i = 0; running; i++) {
        counter++;
        
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        // Only get new inputs according to the sample rate.
        if (i % sample_rate == 0) {
            // Fetch input.
            for (field_size_t j = 0; j < inputs_count; j++) {
                inputs[j] = 1 + (rand() % (sample_rate - 1));
            }
            samples_count = 0;
        }

        // Feed the field.
        if (feeding) {
            for (field_size_t k = 0; k < inputs_count; k++) {
                if (samples_count % inputs[k]) {
                    prev_field->neurons[k].value += NEURON_CHARGE_RATE;
                }
            }
        }

        if (counter % plotInterval == 0) {
            // Clear the window with black color.
            tex->clear(sf::Color(31, 31, 31, 255));

            // Highlight input neurons.
            for (field_size_t i = 0; i < inputs_count; i++) {
                sf::CircleShape neuronCircle;
                
                float radius = 6.0f;

                neuronCircle.setRadius(radius);
                neuronCircle.setOutlineThickness(1);
                neuronCircle.setFillColor(sf::Color::Transparent);
                neuronCircle.setOutlineColor(sf::Color(255, 255, 255, 64));
                neuronCircle.setPosition(xNeuronPositions[i] * textureWidth, yNeuronPositions[i] * textureHeight);
                neuronCircle.setOrigin(radius, radius);

                tex->draw(neuronCircle);
            }

            // Draw neurons.
            drawNeurons(next_field, tex, textureWidth, textureHeight, xNeuronPositions, yNeuronPositions, font);

            // Draw synapses.
            drawSynapses(next_field, tex, textureWidth, textureHeight, xNeuronPositions, yNeuronPositions);

            // End the current frame.
            plot(tex);
            
            usleep(5000);
        }

        // Tick the field.
        f2d_tick(prev_field, next_field, 0x0040u);

        samples_count ++;
    }
    return 0;
}
