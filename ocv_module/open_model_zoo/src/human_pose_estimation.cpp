#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/human_pose_estimation.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

#include "human_pose.hpp"
#include "render_human_pose.hpp"

#ifdef HAVE_INF_ENGINE
#include "human_pose_estimator.hpp"
#endif

#ifdef HAVE_OPENCV_DNN
#include "opencv2/open_model_zoo/dnn.hpp"
#include "opencv2/dnn.hpp"
#endif

using namespace human_pose_estimation;

namespace cv { namespace open_model_zoo {

struct HumanPoseEstimationImpl::Impl
{
    Impl(const Topology& t, std::string device)
    {
#ifdef HAVE_INF_ENGINE
        if (t.getOriginFramework() == "dldt")
        {
            std::string configPath = t.getConfigPath();
            device = device == "GPU16" ? "GPU" : device;
            estimator.reset(new HumanPoseEstimator(configPath, device, false));
        }
        else
#endif
        {
#ifdef HAVE_OPENCV_DNN
            dnnNet = DnnModel(t);
            dnnNet->setPreferableTarget(strToDnnTarget(device));
#endif
        }
    }

    void process(Mat frame, std::vector<HumanPose>& poses)
    {
#ifdef HAVE_INF_ENGINE
        if (!estimator.empty())
            poses = estimator->estimate(frame);
#endif

#ifdef HAVE_OPENCV_DNN
        if (!dnnNet.empty())
        {
            std::vector<Mat> outs;
            dnnNet->predict(frame, outs);
            CV_Assert(outs.size() == 1);

            float* heatMapsData = outs[0].ptr<float>(0, 0);
            int mapOffset = outs[0].size[2] * outs[0].size[3];

            // COCO pose
            float* pafsData = outs[0].ptr<float>(0, 19);
            poses = postprocess(heatMapsData, mapOffset, 18,
                                pafsData, mapOffset, 38,
                                outs[0].size[3], outs[0].size[2],
                                frame.size(), Vec4i(), 8, 4);
        }
#endif
    }

#ifdef HAVE_INF_ENGINE
    Ptr<HumanPoseEstimator> estimator;
#endif

#ifdef HAVE_OPENCV_DNN
    Ptr<dnn::Model> dnnNet;
#endif
};

HumanPoseEstimation::HumanPoseEstimationImpl(const std::string& device)
{
    Topology t;
#ifdef HAVE_INF_ENGINE
    if (device == "GPU16" || device == "MYRIAD")
        t = topologies::human_pose_estimation_fp16();
    else
        t = topologies::human_pose_estimation();
#elif defined(HAVE_OPENCV_DNN)
    t = openpose_coco();
#else
    CV_Error(Error::StsError, "OpenCV or Inference Engine is required");
#endif
    impl.reset(new Impl(t, device));
}

HumanPoseEstimation::HumanPoseEstimationImpl(const Topology& t, const std::string& device)
    : impl(new Impl(t, device))
{
}

void HumanPoseEstimation::process(InputArray frame, CV_OUT std::vector<Pose>& humanPoses)
{
    std::vector<HumanPose> poses;
    impl->process(frame.getMat(), poses);

    humanPoses.resize(poses.size());
    for (size_t i = 0; i < poses.size(); ++i)
    {
        humanPoses[i].keypoints = poses[i].keypoints;
        humanPoses[i].type = "COCO";
    }
}

void HumanPoseEstimation::render(InputOutputArray frame, const std::vector<Pose>& humanPoses)
{
    Mat img = frame.getMat();

    std::vector<HumanPose> poses(humanPoses.size());
    for (size_t i = 0; i < poses.size(); ++i)
    {
        poses[i].keypoints = humanPoses[i].keypoints;
    }
    renderHumanPose(poses, img);
}

}}  // namespace cv::open_model_zoo
