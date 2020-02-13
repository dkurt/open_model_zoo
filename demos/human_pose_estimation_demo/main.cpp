// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <chrono>
#include <thread>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        cv::namedWindow("ICV Human Pose Estimation", cv::WINDOW_NORMAL);

        std::queue<cv::Mat> framesQueue;
        std::queue<std::vector<HumanPose> > posesQueue;
        std::mutex framesQueueMutex;

        double inferenceTime = 0.0;

        std::thread drawingThread([&](){
            while (true) {
                cv::Mat image;
                std::vector<HumanPose> poses;
                std::stringstream fpsSs;

                framesQueueMutex.lock();
                if (!framesQueue.empty())
                {
                    image = framesQueue.front();
                    poses = posesQueue.front();
                    framesQueue.pop();
                    posesQueue.pop();
                }
                else
                {
                    framesQueueMutex.unlock();
                    continue;
                }
                framesQueueMutex.unlock();

                renderHumanPose(poses, image);

                cv::Mat fpsPane(35, 155, CV_8UC3);
                fpsPane.setTo(cv::Scalar(153, 119, 76));
                // cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
                // cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
                fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
                // cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                //             cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
                cv::imshow("ICV Human Pose Estimation", image);

                if (cv::waitKey(1) == 27) {
                    break;
                }
            }
        });

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        cv::VideoCapture cap;
        cv::resizeWindow("ICV Human Pose Estimation", 1280, 960);
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        int delay = 33;

        // read input (video) frame
        cv::Mat curr_frame; cap >> curr_frame;
        cv::Mat next_frame;
        if (!cap.grab()) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        estimator.reshape(curr_frame);  // Do not measure network reshape, if it happened

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key" << std::endl;
            std::cout << "To pause execution, switch to the output window and press 'p' key" << std::endl;
        }
        std::cout << std::endl;

        cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
        Presenter presenter(FLAGS_u, static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) - graphSize.height - 10, graphSize);
        std::vector<HumanPose> poses;
        bool isLastFrame = false;
        bool isAsyncMode = true; // execution is always started in SYNC mode
        bool isModeChanged = false; // set to true when execution mode is changed (SYNC<->ASYNC)
        bool blackBackground = FLAGS_black;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double decode_time = 0, render_time = 0;

        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();
            //here is the first asynchronus point:
            //in the async mode we capture frame to populate the NEXT infer request
            //in the regular mode we capture frame to the current infer request

            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true; //end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    estimator.frameToBlobCurr(curr_frame);
                }
                if (!isLastFrame) {
                    estimator.frameToBlobNext(next_frame);
                }
            } else if (!isModeChanged) {
                estimator.frameToBlobCurr(curr_frame);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the trully Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    estimator.startCurr();
                }
                if (!isLastFrame) {
                    estimator.startNext();
                }
            } else if (!isModeChanged) {
                estimator.startCurr();
            }

            if (estimator.readyCurr()) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);
                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();

                if (!FLAGS_no_show) {
                    if (blackBackground) {
                        curr_frame = cv::Mat::zeros(curr_frame.size(), curr_frame.type());
                    }
                    std::ostringstream out;
                    out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                        << (decode_time + render_time) << " ms";

                    // cv::putText(curr_frame, out.str(), cv::Point2f(0, 25),
                    //             cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                    out.str("");
                    out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                    out << std::fixed << std::setprecision(2) << wall.count()
                        << " ms (" << 1000.f / wall.count() << " fps)";
                    // cv::putText(curr_frame, out.str(), cv::Point2f(0, 50),
                    //             cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                    if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                        out.str("");
                        out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                        // cv::putText(curr_frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                        //     cv::Scalar(255, 0, 0));
                    }
                }

                poses = estimator.postprocessCurr();

                if (FLAGS_r) {
                    if (!poses.empty()) {
                        std::time_t result = std::time(nullptr);
                        std::cout << std::asctime(std::localtime(&result));
                     }

                    for (HumanPose const& pose : poses) {
                        std::stringstream rawPose;
                        rawPose << std::fixed << std::setprecision(0);
                        for (auto const& keypoint : pose.keypoints) {
                            rawPose << keypoint.x << "," << keypoint.y << " ";
                        }
                        rawPose << pose.score;
                        std::cout << rawPose.str() << std::endl;
                    }
                }

                if (!FLAGS_no_show) {
                    framesQueueMutex.lock();
                    framesQueue.push(curr_frame.clone());
                    posesQueue.push(poses);
                    framesQueueMutex.unlock();
                }
            }

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                estimator.swapRequest();
            }

            const int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (27 == key) { // Esc
                break;
            } else if (9 == key) { // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            } else if (32 == key) { // Space
                blackBackground ^= true;
            }
            presenter.handleKey(key);
        }

        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;
        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
