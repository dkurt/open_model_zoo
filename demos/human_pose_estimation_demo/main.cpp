// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <windows.h>

#include <inference_engine.hpp>

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
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    cv::TickMeter rWristUpTime, lWristUpTime;
    bool rWristUp = false, lWristUp = false;
    rWristUpTime.start();
    lWristUpTime.start();

    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        int delay = 33;
        double inferenceTime = 0.0;
        cv::Mat image;
        if (!cap.read(image)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        estimator.estimate(image);  // Do not measure network reshape, if it happened
        do {
            double t1 = cv::getTickCount();
            std::vector<HumanPose> poses = estimator.estimate(image);
            double t2 = cv::getTickCount();
            if (inferenceTime == 0) {
                inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
            } else {
                inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
            }
            if (FLAGS_r) {
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

            if (FLAGS_no_show) {
                continue;
            }

            if (!poses.empty()) {
                cv::Point2f nose = poses[0].keypoints[0];

                cv::Point2f rWrist = poses[0].keypoints[4];
                if (rWrist.y != -1 && rWrist.y < nose.y) {
                    if (!rWristUp) {
                        INPUT input;
                        input.type = INPUT_KEYBOARD;
                        input.ki.wVk = VK_RIGHT;

                        input.ki.dwFlags = 0;
                        SendInput(1, &input, sizeof(INPUT));

                        cv::waitKey(50);

                        input.ki.dwFlags = KEYEVENTF_KEYUP;
                        SendInput(1, &input, sizeof(INPUT));
                    }
                    rWristUp = true;
                    rWristUpTime.reset();
                    rWristUpTime.start();
                }
                else
                {
                    rWristUpTime.stop();
                    if (rWristUpTime.getTimeSec() > 1)
                        rWristUp = false;
                    rWristUpTime.start();
                }

                cv::Point2f lWrist = poses[0].keypoints[7];
                if (lWrist.y != -1 && lWrist.y < nose.y) {
                    if (!lWristUp) {
                        INPUT input;
                        input.type = INPUT_KEYBOARD;
                        input.ki.wVk = VK_LEFT;

                        input.ki.dwFlags = 0;
                        SendInput(1, &input, sizeof(INPUT));

                        cv::waitKey(50);

                        input.ki.dwFlags = KEYEVENTF_KEYUP;
                        SendInput(1, &input, sizeof(INPUT));
                    }
                    lWristUp = true;
                    lWristUpTime.reset();
                    lWristUpTime.start();
                }
                else
                {
                    lWristUpTime.stop();
                    if (lWristUpTime.getTimeSec() > 1)
                        lWristUp = false;
                    lWristUpTime.start();
                }
            }

            // renderHumanPose(poses, image);
            //
            // cv::Mat fpsPane(35, 155, CV_8UC3);
            // fpsPane.setTo(cv::Scalar(153, 119, 76));
            // cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            // cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
            // std::stringstream fpsSs;
            // fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            // cv::putText(image, fpsSs.str(), cv::Point(16, 32),
            //             cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
            // cv::imshow("ICV Human Pose Estimation", image);
            // cv::waitKey(1);
        } while (cap.read(image));
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
