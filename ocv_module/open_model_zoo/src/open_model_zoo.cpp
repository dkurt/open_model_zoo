#include "opencv2/open_model_zoo.hpp"

#include <iostream>
#include <sstream>

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>

#define path(cache, file) utils::fs::canonical(utils::fs::join(cache, file))

namespace cv { namespace open_model_zoo {

struct Topology::Impl
{
    std::string modelURL, modelSHA, modelPath,
                configURL, configSHA, configPath,
                archiveURL, archiveSHA, archivePath,
                description, license, topologyName, framework;
    std::map<std::string, std::string> modelOptimizerArgs;
};

Topology::Topology() {}

Topology::Topology(const std::map<std::string, std::string>& config)
    : impl(new Impl())
{
    impl->topologyName = config.at("topology_name");
    auto cache = utils::fs::getCacheDirectory("open_model_zoo_cache",
                                              "OPENCV_OPEN_MODEL_ZOO_CACHE_DIR");
    cache = utils::fs::join(cache, impl->topologyName);

    auto it = config.find("model_optimizer_args");
    if (it != config.end())
    {
        std::stringstream s(it->second);
        std::string arg;
        while (s >> arg)
        {
            int delim = arg.find('=');
            impl->modelOptimizerArgs[arg.substr(0, delim)] = arg.substr(delim + 1);
        }
    }

    it = config.find("archive_url");
    if (it != config.end())
    {
        static const std::string kDownloadDir = "$dl_dir";

        impl->archiveURL = config.at("archive_url");
        impl->archiveSHA = config.at("archive_sha256");
        impl->archivePath = path(cache, config.at("archive_name"));
        impl->configURL = impl->modelURL = impl->archiveURL;
        impl->configSHA = impl->modelSHA = impl->archiveSHA;

        it = impl->modelOptimizerArgs.find("--input_model");
        if (it == impl->modelOptimizerArgs.end())
            CV_Error(Error::StsNotImplemented, "Expected --input_model option");

        // Weights path is in MO arguments
        impl->modelPath = it->second;
        if (impl->modelPath.substr(0, kDownloadDir.size()) == kDownloadDir)
            impl->modelPath = impl->modelPath.substr(kDownloadDir.size() + 1);

        // Config path is in MO arguments
        it = impl->modelOptimizerArgs.find("--input_proto");
        if (it != impl->modelOptimizerArgs.end())
        {
            impl->configPath = it->second;
            if (impl->configPath.substr(0, kDownloadDir.size()) == kDownloadDir)
                impl->configPath = impl->configPath.substr(kDownloadDir.size() + 1);
        }
    }
    else
    {
        it = config.find("model_url");
        if (it != config.end())
        {
            impl->modelURL = config.at("model_url");
            impl->modelSHA = config.at("model_sha256");
            impl->modelPath = config.at("model_path");
        }

        it = config.find("config_url");
        if (it != config.end())
        {
            impl->configURL = config.at("config_url");
            impl->configSHA = config.at("config_sha256");
            impl->configPath = config.at("config_path");
        }
    }

    impl->description = config.at("description");
    impl->license = config.at("license");
    impl->framework = config.at("framework");

    if (!impl->modelPath.empty())
        impl->modelPath = path(cache, impl->modelPath);
    if (!impl->configPath.empty())
        impl->configPath = path(cache, impl->configPath);
}

std::string Topology::getDescription() const { return impl->description; }
std::string Topology::getLicense() const { return impl->license; }
std::string Topology::getConfigPath() const { return impl->configPath; }
std::string Topology::getModelPath() const { return impl->modelPath; }
std::string Topology::getName() const { return impl->topologyName; }
std::string Topology::getOriginFramework() const { return impl->framework; }

void Topology::getModelInfo(std::string& url, std::string& sha, std::string& path) const
{
    url = impl->modelURL;
    sha = impl->modelSHA;
    path = impl->modelPath;
}

void Topology::getConfigInfo(String& url, String& sha, String& path) const
{
    url = impl->configURL;
    sha = impl->configSHA;
    path = impl->configPath;
}

void Topology::getArchiveInfo(String& url, String& sha, String& path) const
{
    url = impl->archiveURL;
    sha = impl->archiveSHA;
    path = impl->archivePath;
}

void Topology::getMeans(std::map<std::string, Scalar>& means) const
{
    // mean_values has format "name1[v1,v2,v3] name2[v1,v2,v3]"
    means.clear();
    auto it = impl->modelOptimizerArgs.find("--mean_values");
    if (it == impl->modelOptimizerArgs.end())
        return;

    std::string arg, name, value;
    std::stringstream args(it->second);
    while (args >> arg)
    {
        int start = arg.find('[');
        int end = arg.find(']');
        name = arg.substr(0, start);
        std::stringstream ss(arg.substr(start + 1, end - 1));

        Scalar mean;
        int i = 0;
        while (std::getline(ss, value, ','))
        {
            std::stringstream(value) >> mean[i];
            i += 1;
        }
        arg = arg.substr(end + 1);

        means[name] = mean;
    }
}

void Topology::getScales(std::map<std::string, double>& scales) const
{
    // mean_values has format "name1[value1] name2[value2]"
    scales.clear();
    auto it = impl->modelOptimizerArgs.find("--scale_values");
    if (it == impl->modelOptimizerArgs.end())
        return;

    std::string arg, name;
    std::stringstream args(it->second);
    double value;
    while (args >> arg)
    {
        int start = arg.find('[');
        int end = arg.find(']');
        name = arg.substr(0, start);
        std::stringstream(arg.substr(start + 1, end - 1)) >> value;
        scales[name] = value;
    }
}

void Topology::getInputShape(std::vector<int>& shape) const
{
    // input_shape has format "[v1,v2,v3,v4]"
    shape.clear();

    auto it = impl->modelOptimizerArgs.find("--input_shape");
    if (it == impl->modelOptimizerArgs.end())
        return;

    std::string values = it->second;
    values = values.substr(values.find('[') + 1, values.find(']'));

    int value;
    std::string valueStr;
    std::stringstream ss(values);
    while (std::getline(ss, valueStr, ','))
    {
        std::stringstream(valueStr) >> value;
        shape.push_back(value);
    }
}

std::map<String, String> Topology::getModelOptimizerArgs() const
{
    return impl->modelOptimizerArgs;
}

void Topology::download() const
{
    std::vector<std::string> paths(2);
    std::vector<std::string> urls(2);
    std::string sha;
    getModelInfo(urls[0], sha, paths[0]);
    getConfigInfo(urls[1], sha, paths[1]);
    for (int i = 0; i < 2; ++i)
    {
        if (!paths[i].empty() && !utils::fs::exists(paths[i]))
            CV_LOG_WARNING(NULL, "\nFile " + paths[i] + " does not exist. Download it from " +
                                 urls[i] + " or specify OPENCV_OPEN_MODEL_ZOO_CACHE_DIR environment variable");
    }
}

}}  // namespace cv::omz
