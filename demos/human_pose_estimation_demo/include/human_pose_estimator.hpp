// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/core.hpp>

#include "human_pose.hpp"

namespace human_pose_estimation {

std::vector<HumanPose> postprocess(
        const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
        const float* pafsData, const int pafOffset, const int nPafs,
        const int featureMapWidth, const int featureMapHeight,
        const cv::Size& imageSize, const cv::Vec4i& pad,
        int stride, int upsampleRatio);

class HumanPoseEstimator {
public:
    static const size_t keypointsNumber;

    HumanPoseEstimator(const std::string& modelPath,
                       const std::string& targetDeviceName,
                       bool enablePerformanceReport = false);
    std::vector<HumanPose> estimate(const cv::Mat& image);
    ~HumanPoseEstimator();

private:
    void preprocess(const cv::Mat& image, float* buffer) const;
    bool inputWidthIsChanged(const cv::Size& imageSize);

    int stride;
    cv::Vec4i pad;
    cv::Vec3f meanPixel;
    cv::Size inputLayerSize;
    int upsampleRatio;
    InferenceEngine::Core ie;
    std::string targetDeviceName;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest request;
    InferenceEngine::CNNNetReader netReader;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    bool enablePerformanceReport;
    std::string modelPath;
};
}  // namespace human_pose_estimation
