#ifndef __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__

#include "opencv2/core.hpp"

#include <map>

namespace cv { namespace open_model_zoo {

    class CV_EXPORTS_W_SIMPLE Topology
    {
    public:
        Topology();

        Topology(const std::map<std::string, std::string>& config);

        /**
         * @brief Convert model from native framework to OpenVINO Intermadiate Representation (IR)
         * @param[out] xmlPath Path to generated .xml file.
         * @param[out] binPath Path to generated .bin file.
         *
         * This method calls Model Optimizer Python tool with arguments from
         * topology's description. If model is already in IR format or there are
         * already converted files - returns paths without MO invocation.
         * @note Method is available from Python3 only.
         */
        CV_WRAP void convertToIR(CV_OUT String& xmlPath, CV_OUT String& binPath) const;

        /**
         * @brief Get a name of topology.
         *
         * Name is generated from description file. Symbols '-' and '.' are replaced to '_'
         *
         * OpenVINO IR models may additionally have precision prefix which specifies
         * weights precision (FP32 models have no prefix). In example,
         * `face_detection_retail_0004` for FP32, `face_detection_retail_0004_fp16` for FP16
         *
         * OpenVINO IR models with version digits have aliases. In example, there is
         * `face_detection_retail` for `face_detection_retail_0004`. If there are
         * multiple versions, alias is generated for the highest version:
         * `face_detection_retail_0004` and `face_detection_retail_0005` but
         * `face_detection_retail` is the same as `face_detection_retail_0005`.
         */
        CV_WRAP String getName() const;

        CV_WRAP void download();

        CV_WRAP std::string getDescription() const;

        CV_WRAP std::string getLicense() const;

        CV_WRAP void getArchiveInfo(CV_OUT String& url, CV_OUT String& sha256,
                                    CV_OUT String& path) const;

        CV_WRAP void getModelInfo(CV_OUT String& url, CV_OUT String& sha256,
                                  CV_OUT String& path) const;

        CV_WRAP void getConfigInfo(CV_OUT String& url, CV_OUT String& sha256,
                                   CV_OUT String& path) const;

        CV_WRAP String getModelPath() const;

        CV_WRAP String getConfigPath() const;

        CV_WRAP String getOriginFramework() const;

        void getMeans(std::map<std::string, Scalar>& means) const;

        void getScales(std::map<std::string, double>& scales) const;

        CV_WRAP void getInputShape(CV_OUT std::vector<int>& shape) const;

        std::map<String, String> getModelOptimizerArgs() const;

    protected:
        struct Impl;
        Ptr<Impl> impl;
    };

    class CV_EXPORTS_W TextRecognitionPipelineImpl
    {
    public:
        CV_WRAP TextRecognitionPipelineImpl(const Topology& detection,
                                            const Topology& recognition);

        CV_WRAP void process(InputArray frame, CV_OUT std::vector<RotatedRect>& rects,
                             CV_OUT std::vector<String>& texts);

    private:
        struct Impl;
        Ptr<Impl> impl;
    };

    typedef TextRecognitionPipelineImpl TextRecognitionPipeline;

#if 0
    // This is a trick to enable open_model_zoo::TextRecognitionPipeline both in Python and in C++
    CV_WRAP_AS(TextRecognitionPipeline) Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const Topology& detection, const Topology& recognition);
#endif

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
