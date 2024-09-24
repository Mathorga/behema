#include <behema/behema.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

int main(int argc, char **argv) {
    bhm_cortex_size_t cortex_width = 100;
    bhm_cortex_size_t cortex_height = 60;
    bhm_nh_radius_t nh_radius = 2;
    bhm_ticks_count_t sampleWindow = BHM_SAMPLE_WINDOW_MID;
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

    // Create network model.
    bhm_cortex2d_t even_cortex;
    bhm_cortex2d_t odd_cortex;
    bhm_error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    c2d_set_sample_window(&even_cortex, sampleWindow);
    c2d_set_evol_step(&even_cortex, 0x01U);
    c2d_set_pulse_mapping(&even_cortex, BHM_PULSE_MAPPING_FPROP);
    c2d_set_max_syn_count(&even_cortex, 24);
    c2d_copy(&odd_cortex, &even_cortex);

    int counter = 0;

    // Inputs.
    bhm_input2d_t leftEye;
    i2d_init(&leftEye, 0, 0, (cortex_width / 10) * 3, 1, BHM_DEFAULT_EXC_VALUE * 4, BHM_PULSE_MAPPING_FPROP);

    bhm_input2d_t rightEye;
    i2d_init(&rightEye, (cortex_width / 10) * 7, 0, cortex_width, 1, BHM_DEFAULT_EXC_VALUE * 4, BHM_PULSE_MAPPING_FPROP);

    cv::Size eyeSize = cv::Size(leftEye.x1 - leftEye.x0, leftEye.y1 - leftEye.y0);

    // bhm_cortex_size_t lTimedInputsCoords[] = {0, cortex_height - 5, 1, cortex_height};
    // bhm_cortex_size_t rTimedInputsCoords[] = {cortex_width - 1, cortex_height - 5, cortex_width, cortex_height};

    char touchFileName[40];
    char inhexcFileName[40];
    sprintf(touchFileName, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    sprintf(inhexcFileName, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);

    c2d_touch_from_map(&even_cortex, touchFileName);
    c2d_inhexc_from_map(&even_cortex, inhexcFileName);

    bhm_ticks_count_t samplingBound = sampleWindow - 1;
    bhm_ticks_count_t sample_step = samplingBound;

    for (int i = 0; ; i++) {
        counter++;

        bhm_cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        bhm_cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        if (i % 1000 == 0) {
            printf("\n%d: saved file\n", i);
            c2d_to_file(prev_cortex, "./out/cortex.c2d");
        }

        // Only get new inputs according to the sample rate.
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
            for (bhm_cortex_size_t y = 0; y < eyeSize.height; y++) {
                for (bhm_cortex_size_t x = 0; x < eyeSize.width; x++) {
                    cv::Vec3b val = resized.at<cv::Vec3b>(cv::Point(x, y));
                    leftEye.values[IDX2D(x, y, eyeSize.width)] = fmap(val[2],
                                                                         0, 255,
                                                                         0, samplingBound);
                    rightEye.values[IDX2D(x, y, eyeSize.width)] = fmap(val[0],
                                                                         0, 255,
                                                                         0, samplingBound);
                }
            }

            sample_step = 0;
        }

        // Feed the cortex.
        c2d_feed2d(prev_cortex, &leftEye);
        c2d_feed2d(prev_cortex, &rightEye);

        // c2d_sample_sqfeed(prev_cortex, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, BHM_DEFAULT_EXC_VALUE * 4);
        // c2d_sample_sqfeed(prev_cortex, bInputsCoords[0], bInputsCoords[1], bInputsCoords[2], bInputsCoords[3], sample_step, bInputs, BHM_DEFAULT_EXC_VALUE * 4);

        // c2d_sqfeed(prev_cortex, lTimedInputsCoords[0], lTimedInputsCoords[1], lTimedInputsCoords[2], lTimedInputsCoords[3], BHM_DEFAULT_EXC_VALUE / 3);
        // c2d_sqfeed(prev_cortex, rTimedInputsCoords[0], rTimedInputsCoords[1], rTimedInputsCoords[2], rTimedInputsCoords[3], BHM_DEFAULT_EXC_VALUE / 3);

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);

        // usleep(10000);

        sample_step++;
    }
    
    return 0;
}