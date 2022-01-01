#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <portia/portia.h>

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    double slope = ((double) output_end - (double) output_start) / ((double) input_end - (double) input_start);
    return (double) output_start + slope * ((double) input - (double) input_start);
}

int main(int argc, char **argv) {
    cv::Mat frame;
    cv::VideoCapture cap;
    char* fileName;

    // Input handling.
    switch (argc) {
        case 2:
            fileName = argv[1];
            break;
        default:
            printf("USAGE: video </input/file/path>\n");
            exit(0);
            break;
    }

    cap.open(fileName);
    if (!cap.isOpened()) {
        printf("ERROR! Unable to open video file\n");
        return -1;
    }

    srand(time(NULL));

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    cortex_size_t cortex_width = 100;
    cortex_size_t cortex_height = 60;
    nh_radius_t nh_radius = 2;
    ticks_count_t sampleWindow = 10;
    ticks_count_t samplingBound = sampleWindow - 1;
    error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    c2d_set_evol_step(&even_cortex, 0x0AU);
    c2d_set_pulse_window(&even_cortex, 0x3AU);
    c2d_set_syngen_pulses_count(&even_cortex, 0x01U);
    c2d_set_max_touch(&even_cortex, 0.3F);
    c2d_set_sample_window(&even_cortex, sampleWindow);
    c2d_set_pulse_mapping(&even_cortex, PULSE_MAPPING_FPROP);
    c2d_set_inhexc_ratio(&even_cortex, 0x0FU);
    odd_cortex = *c2d_copy(&even_cortex);

    // Coordinates for input neurons.
    cortex_size_t lInputsCoords[] = {cortex_width / 4, cortex_height / 4, (cortex_width / 4) * 3, (cortex_height / 4) * 3};
    ticks_count_t* lInputs = (ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(ticks_count_t));

    cv::Size inputSize = cv::Size(lInputsCoords[2] - lInputsCoords[0], lInputsCoords[3] - lInputsCoords[1]);

    ticks_count_t sample_step = samplingBound;

    int frameCount = 0;

    for (int i = 0; ; i++) {
        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;


        if (sample_step > samplingBound) {
            // Fetch input.
            bool validFrame = cap.read(frame);
            // cap >> frame;

            if (!validFrame) {
                // Write cortex to file and exit.
                c2d_to_file(&even_cortex, (char *) "out/video.c2d");
                break;
            }

            cv::Mat resized;
            cv::resize(frame, resized, inputSize);
            
            cv::cvtColor(resized, frame, cv::COLOR_BGR2GRAY);

            resized.at<uint8_t>(cv::Point(0, 0));
            for (cortex_size_t y = 0; y < lInputsCoords[3] - lInputsCoords[1]; y++) {
                for (cortex_size_t x = 0; x < lInputsCoords[2] - lInputsCoords[0]; x++) {
                    uint8_t val = frame.at<uint8_t>(cv::Point(x, y));
                    lInputs[IDX2D(x, y, lInputsCoords[2] - lInputsCoords[0])] = map(val,
                                                                                    0, 255,
                                                                                    0, even_cortex.sample_window - 1);
                }
            }
            sample_step = 0;

            // cv::resize(frame, resized, inputSize * 10, 0, 0, cv::INTER_NEAREST);
            // cv::imshow("Result", resized);

            // if (cv::waitKey(1) >= 0) {
            //     // Write cortex to file and exit.
            //     c2d_to_file(&even_cortex, (char *) "out/video.c2d");
            //     break;
            // }

            if (frameCount % 100 == 0) {
                printf("Frame %d\n", frameCount);
            }

            frameCount++;
        }

        // Feed the cortex.
        c2d_sample_sqfeed(prev_cortex, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, DEFAULT_EXCITING_VALUE);

        sample_step++;

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}