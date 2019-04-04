// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <thread>

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
                cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
                cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
                fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
                cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                            cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
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
            framesQueueMutex.lock();
            framesQueue.push(image);
            posesQueue.push(poses);
            framesQueueMutex.unlock();
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
