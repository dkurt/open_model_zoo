// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>

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

static void drawLightning(cv::Mat& image, cv::Point start, cv::Point end, float splitProb, int depth)
{
    if (depth == 2)
        return;

    const int numSteps = 5;
    cv::Point lastPoint = start;
    cv::Point dir = end - lastPoint;
    float step = 1.0 / numSteps;
    for (int i = 0; i < numSteps; ++i)
    {

        cv::Mat r(1, 1, CV_32F);
        cv::randu(r, -30, 30);
        cv::Mat m = cv::getRotationMatrix2D(lastPoint, r.at<float>(0, 0), 1.0);

        cv::Mat dst(3, 1, CV_64F);
        dst.at<double>(0, 0) = lastPoint.x + dir.x * step;
        dst.at<double>(0, 1) = lastPoint.y + dir.y * step;
        dst.at<double>(0, 2) = 1;

        dst = m * dst;

        cv::Point res;
        res.x = dst.at<double>(0, 0);
        res.y = dst.at<double>(1, 0);

        cv::line(image, lastPoint, res, cv::Scalar(0, 255, 255), 2);
        lastPoint = res;

        // Left
        cv::randu(r, 0, 1);
        if (r.at<float>(0, 0) < splitProb)
        {
            cv::Point branchEnd;
            cv::randu(r, std::min(end.x, lastPoint.x), std::max(end.x, lastPoint.x));
            branchEnd.x = r.at<float>(0, 0);

            cv::randu(r, std::min(end.y, lastPoint.y), std::max(end.y, lastPoint.y));
            branchEnd.y = r.at<float>(0, 0);

            drawLightning(image, lastPoint, branchEnd, splitProb * 0.7, depth + 1);
        }
    }
    cv::line(image, lastPoint, end, cv::Scalar(0, 255, 255), 2);
}

int main(int argc, char* argv[]) {
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
        if (!FLAGS_no_show) {
            std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        }

        cv::Mat hammer = cv::imread("/home/dkurtaev/open_model_zoo/hammer.png");
        cv::Mat mask;
        do {
            double t1 = static_cast<double>(cv::getTickCount());
            std::vector<HumanPose> poses = estimator.estimate(image);
            double t2 = static_cast<double>(cv::getTickCount());
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

            // renderHumanPose(poses, image);

            if (mask.empty())
                mask = cv::Mat::zeros(image.size(), CV_8UC3);
            else
                mask.setTo(0);

            for (HumanPose const& pose : poses) {
              cv::Point2f wrist = pose.keypoints[4];
              cv::Point2f elbow = pose.keypoints[3];
              cv::Point2f neck = pose.keypoints[1];
              cv::Point2f rhip = pose.keypoints[8];
              cv::Point2f lhip = pose.keypoints[11];
              cv::Point2f rshoulder = pose.keypoints[2];
              cv::Point2f lshoulder = pose.keypoints[5];

              if ((wrist.x == -1 && wrist.y == -1) ||
                  (neck.x == -1 && neck.y == -1) ||
                  (lhip.x == -1 && lhip.y == -1 && rhip.x == -1 && rhip.y == -1 &&
                   rshoulder.x == -1 && rshoulder.y == -1 && lshoulder.x == -1 && lshoulder.y == -1))
                  continue;

              cv::Point2f dir = wrist - elbow;
              float length = std::sqrt(dir.dot(dir));
              float offset = 0.25 * length;
              dir.y /= length;
              dir.x /= length;
              float angle = std::atan2(dir.y, dir.x) * 180 / M_PI;

              float hammerSize;
              if (lhip.x != -1 || rhip.x != -1)
              {
                  cv::Point2f bodyDir = (lhip.x != -1 ? lhip : rhip) - neck;
                  float bodyLength = std::sqrt(bodyDir.dot(bodyDir));
                  hammerSize = 0.9 * bodyLength;
              }
              else
              {
                  cv::Point2f shoulderDir = (lshoulder.x != -1 ? lshoulder : rshoulder) - neck;
                  float shoulderLength = std::sqrt(shoulderDir.dot(shoulderDir));
                  hammerSize = 2.25 * shoulderLength;
              }

              float scale = hammerSize / hammer.cols;
              cv::Mat m = cv::getRotationMatrix2D(cv::Point(hammer.cols / 2, hammer.rows / 2), -angle+90, scale);
              m.at<double>(0, 2) = wrist.x - 0.6 * hammerSize * dir.y - dir.x * (0.5 * hammerSize * hammer.rows / hammer.cols - offset);
              m.at<double>(1, 2) = wrist.y + 0.6 * hammerSize * dir.x - dir.y * (0.5 * hammerSize * hammer.rows / hammer.cols - offset);
              cv::warpAffine(hammer, mask, m, mask.size());
              mask.copyTo(image, mask);

              cv::Point2f nose = pose.keypoints[0];
              if (wrist.y < nose.y)
              {
                  cv::Point dst;
                  dst.x = wrist.x - 0.6 * hammerSize * dir.y;
                  dst.y = wrist.y + 0.6 * hammerSize * dir.x;
                  // Generate a tree
                  drawLightning(image, cv::Point(elbow.x, 0), dst, 0.9, 0);
              }
            }

            // cv::Mat fpsPane(35, 155, CV_8UC3);
            // fpsPane.setTo(cv::Scalar(153, 119, 76));
            // cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            // cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
            // std::stringstream fpsSs;
            // fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            // cv::putText(image, fpsSs.str(), cv::Point(16, 32),
            //             cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
            static int idx = 0;
            cv::imshow("ICV Human Pose Estimation", image);
            cv::imwrite(cv::format("images/%06d.jpg", idx), image);
            idx += 1;

            int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (key == 27) {
                break;
            }
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
