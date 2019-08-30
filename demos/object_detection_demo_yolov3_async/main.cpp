// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov3_async/main.cpp
* \example object_detection_demo_yolov3_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <thread>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "object_detection_demo_yolov3_async.hpp"

#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    cv::TickMeter tm;
    std::mutex mutex;
};

void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer->GetParamAsInt("num");
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    try {
        auto mask = layer->GetParamAsInts("mask");
        num = mask.size();

        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    } catch (...) {}

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    try {
        /** This demo covers a certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        // read input (video) frame
        cv::Mat frame;
        cap >> frame;

        const size_t width  = frame.cols;
        const size_t height = frame.rows;

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed to get next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /**Loading extensions to the devices **/

#ifdef WITH_EXTENSIONS
        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from the "extension" folder containing
             * custom CPU layer implementations.
            **/
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }
#endif

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Reading network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Setting batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extracting the model name and loading its weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Reading labels (if specified) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(netReader.getNetwork(), FLAGS_d);

        // --------------------------- 5. Creating infer request -----------------------------------------------
        bool process = true;

        // Frames capturing thread
        QueueFPS<cv::Mat> framesQueue;
        std::thread framesThread([&](){
            cv::Mat frame;
            while (process)
            {
                cap >> frame;
                if (!frame.empty())
                    framesQueue.push(frame.clone());
                else
                    break;
                // To regulate video FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });

        // Processing thread
        QueueFPS<cv::Mat> processedFramesQueue;
        QueueFPS<std::vector<DetectionObject> > predictionsQueue;
        std::thread processingThread([&](){

            std::vector<InferRequest::Ptr> async_infer_requests(FLAGS_async);
            std::vector<int> async_infer_is_free(async_infer_requests.size(), true);
            std::vector<int> async_infer_is_ready(async_infer_requests.size(), false);
            std::vector<int> async_infer_is_fake(async_infer_requests.size(), false);
            for (size_t i = 0; i < async_infer_requests.size(); ++i)
            {
                async_infer_requests[i] = network.CreateInferRequestPtr();

                IInferRequest::Ptr infRequestPtr = *async_infer_requests[i].get();
                infRequestPtr->SetUserData(&async_infer_is_ready[i], 0);
                infRequestPtr->SetCompletionCallback(
                    [](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode)
                    {
                        int* is_ready;
                        request->GetUserData((void**)&is_ready, 0);
                        *is_ready = 1;
                    }
                );
            }

            std::queue<int> started_requests_ids;
            cv::Mat frame;
            while (process)
            {
                bool fakeFrame = true;
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    fakeFrame = false;
                }

                if (frame.empty())
                    continue;

                bool started = false;
                while (!started)
                {
                    //
                    // Check for finished requests and start a new one
                    //
                    for (size_t i = 0; i < async_infer_requests.size(); ++i) {
                        if (async_infer_is_free[i]) {
                            InferRequest::Ptr request = async_infer_requests[i];

                            async_infer_is_free[i] = false;
                            async_infer_is_ready[i] = false;
                            async_infer_is_fake[i] = fakeFrame;
                            if (!fakeFrame)
                                processedFramesQueue.push(frame);  // predictionsQueue is used in rendering

                            FrameToBlob(frame, request, inputName);
                            started_requests_ids.push(i);
                            request->StartAsync();
                            started = true;
                            break;
                        }
                    }

                    //
                    // Postprocess finished requests.
                    //
                    if (started_requests_ids.empty())
                        continue;

                    int request_id = started_requests_ids.front();
                    if (async_infer_is_ready[request_id])
                    {
                        InferRequest::Ptr request = async_infer_requests[request_id];
                        async_infer_is_free[request_id] = true;
                        async_infer_is_ready[request_id] = false;
                        started_requests_ids.pop();

                        // ---------------------------Processing output blobs--------------------------------------------------
                        // Processing results of the CURRENT request
                        const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
                        unsigned long resized_im_h = getTensorHeight(inputDesc);
                        unsigned long resized_im_w = getTensorWidth(inputDesc);
                        std::vector<DetectionObject> objects;
                        // Parsing outputs
                        for (auto &output : outputInfo) {
                            auto output_name = output.first;
                            CNNLayerPtr layer = netReader.getNetwork().getLayerByName(output_name.c_str());
                            Blob::Ptr blob = request->GetBlob(output_name);
                            ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
                        }
                        // Filtering overlapping boxes
                        std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                        for (size_t i = 0; i < objects.size(); ++i) {
                            if (objects[i].confidence == 0)
                                continue;
                            for (size_t j = i + 1; j < objects.size(); ++j)
                                if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                                    objects[j].confidence = 0;
                        }
                        // Adding predictions to the queue
                        std::vector<DetectionObject> detections;
                        for (auto &object : objects) {
                            if (object.confidence > FLAGS_t)
                                detections.push_back(object);
                        }
                        if (!async_infer_is_fake[request_id])
                            predictionsQueue.push(detections);
                        else
                            predictionsQueue.counter += 1;
                    }
                }
            }
        });

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
        std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;

        cv::namedWindow("Detection results", cv::WINDOW_NORMAL);
        // // Uncomment to render video
      	// cv::VideoWriter writer("predictions.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame.size());
        while (cv::waitKey(30) != 27) {
            if (predictionsQueue.empty())
                continue;

            std::vector<DetectionObject> detections = predictionsQueue.get();
            cv::Mat frame = processedFramesQueue.get();

            for (const auto& object : detections) {
                int label = object.class_id;
                /** Drawing only objects when >confidence_threshold probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << object.confidence;
                cv::putText(frame,
                        (label < static_cast<int>(labels.size()) ?
                                labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                            cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                            cv::Scalar(0, 255, 0));
                cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                              cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 255, 0));
            }
            putText(frame, cv::format("video fps: %.2f", framesQueue.getFPS()),
                    cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            putText(frame, cv::format("network fps: %.2f", predictionsQueue.getFPS()),
                    cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            putText(frame, cv::format("render queue: %d", (int)processedFramesQueue.size()),
                    cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            putText(frame, cv::format("%d async requests", (int)FLAGS_async),
                    cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));


            cv::imshow("Detection results", frame);
            // // Uncomment to render video
            // writer << frame;

            // // Uncomment to collect efficiency
            // if (predictionsQueue.counter > 500)
            //     break;
        }
        // // Uncomment to render video
        // writer.release();

        // // Uncomment to collect efficiency
        // std::ofstream outfile("perf.txt", std::ios_base::app);
        // outfile << (int)FLAGS_async << " " << predictionsQueue.getFPS() << std::endl;

        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
