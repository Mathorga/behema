#include <portia/portia.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
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
                 cv::Mat* image,
                 uint32_t textureWidth,
                 uint32_t textureHeight,
                 float* xNeuronPositions,
                 float* yNeuronPositions) {
    for (cortex_size_t i = 0; i < cortex->height; i++) {
        for (cortex_size_t j = 0; j < cortex->width; j++) {
            neuron_t* currentNeuron = &(cortex->neurons[IDX2D(j, i, cortex->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) cortex->fire_threshold);

            float radius = 3.0f;
            cv::Scalar* color;

            if (neuronValue < 0) {
                color = new cv::Scalar(31 - 31 * neuronValue, 15 - 15 * neuronValue, 0);
            } else if (currentNeuron->value > cortex->fire_threshold) {
                color = new cv::Scalar(255, 255, 255, 255);
            } else {
                color = new cv::Scalar(31 + 224 * neuronValue, 15 + 112 * neuronValue, 0);
            }

            cv::circle(*image,
                       cv::Point(xNeuronPositions[IDX2D(j, i, cortex->width)] * textureWidth, yNeuronPositions[IDX2D(j, i, cortex->width)] * textureHeight),
                       radius,
                       *color,
                       cv::FILLED);
        }
    }
}

void drawSynapses(cortex2d_t* cortex,
                  cv::Mat* image,
                  uint32_t textureWidth,
                  uint32_t textureHeight,
                  float* xNeuronPositions,
                  float* yNeuronPositions) {
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
                            cv::line(*image,
                                     cv::Point(xNeuronPositions[neighborIndex] * textureWidth, yNeuronPositions[neighborIndex] * textureHeight),
                                     cv::Point(xNeuronPositions[neuronIndex] * textureWidth, yNeuronPositions[neuronIndex] * textureHeight),
                                     cv::Scalar(127, 63, 15),
                                     1,
                                     cv::LINE_8);
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
    ticks_count_t plotInterval = 1000;

    uint32_t textureWidth = 1366;
    uint32_t textureHeight = 768;

    uint32_t feedingPeriod = (rand() % 1000) * 100;

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
            plotInterval = atoi(argv[5]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    srand(time(0));

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    c2d_set_evol_step(&even_cortex, 0x8AU);
    c2d_set_max_touch(&even_cortex, 0.22F);
    c2d_set_syngen_beat(&even_cortex, 0.1F);
    c2d_set_pulse_window(&even_cortex, 0x10U);
    odd_cortex = *c2d_copy(&even_cortex);

    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));

    initPositions(&even_cortex, xNeuronPositions, yNeuronPositions, false);
    
    // Create the texture to render.
    cv::Mat image = cv::Mat::zeros(textureHeight, textureWidth, CV_8UC3);
    
    bool feeding = true;

    int counter = 0;

    ticks_count_t sample_step = 10;

    cortex_size_t lInputsCoords[] = {5, 5, 30, 20};
    ticks_count_t* lInputs = (ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(ticks_count_t));

    ticks_count_t samples_count = 0;

    bool running = true;

    // Run the program as long as the window is open.
    for (int i = 0; running; i++) {
        counter++;

        // Toggle feeding at feeding period.
        if (counter % feedingPeriod == 0) {
            feeding = !feeding;
            if (feeding) {
                feedingPeriod = (rand() % 1000) * 100;
            } else {
                feedingPeriod = (rand() % 1000) * 10;
            }
            counter = 0;
        }
        
        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        // Only get new inputs according to the sample rate.
        if (i % sample_step == 0) {
            // Fetch input.
            for (cortex_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (cortex_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])] = (rand() % (prev_cortex->sample_window - 1));
                }
            }
            samples_count = 0;
        }

        // Feed the cortex.
        if (feeding) {
            c2d_sample_sqfeed(prev_cortex, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, DEFAULT_EXCITING_VALUE);
        }

        if (counter % plotInterval == 0) {
            // Clear the window with black color.
            image.setTo(cv::Scalar(0, 0, 0));

            // Highlight input neurons.
            for (cortex_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (cortex_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    cv::circle(image,
                            cv::Point(xNeuronPositions[IDX2D(x, y, prev_cortex->width)] * textureWidth, yNeuronPositions[IDX2D(x, y, prev_cortex->width)] * textureHeight),
                            2.0f,
                            cv::Scalar(64, 64, 64),
                            1);
                }
            }

            // Draw synapses.
            drawSynapses(next_cortex, &image, textureWidth, textureHeight, xNeuronPositions, yNeuronPositions);

            // Draw neurons.
            // drawNeurons(next_cortex, &image, textureWidth, textureHeight, xNeuronPositions, yNeuronPositions);

            // Draw input period.
            char periodString[100];
            snprintf(periodString, 100, "Feeding %d - %d", feeding, feedingPeriod);
            cv::putText(image,
                        periodString,
                        cv::Point(30, 30),
                        cv::FONT_HERSHEY_DUPLEX,
                        1.0,
                        CV_RGB(255, 255, 255),
                        1);

            // End the current frame.
            char fileName[100];
            snprintf(fileName, 100, "out/%lu.bmp", (unsigned long) time(NULL));
            cv::imwrite(fileName, image);
            cv::waitKey(5);
            
            usleep(5000);
        }

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);

        samples_count ++;
    }
    return 0;
}
