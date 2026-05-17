#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <optional>
#ifndef MEX_COMPILE_FLAG
#include <opencv2/highgui/highgui_c.h>
#endif

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "../utils/ILoadSaveConfig.h"

#if !defined(cubgs_register)
#define cubgs_register(x) static cuBGS_Register<x> curegister_##x(quote(x))
#endif

namespace bgslibrary {
namespace algorithms {

class cuIBGS : public ILoadSaveConfig {
protected:
  bool first_time = true;
  cv::cuda::GpuMat img_background;
  cv::cuda::GpuMat img_foreground;
  std::string algorithm_name;

public:
  cuIBGS() = default;
  cuIBGS(const std::string _algorithm_name) : algorithm_name(_algorithm_name) {}
  virtual ~cuIBGS() {}

  cv::cuda::GpuMat apply(const cv::cuda::GpuMat &img_input) {
    cv::cuda::GpuMat _img_foreground, _img_background;
    process(img_input, _img_foreground, _img_background);
    _img_background.copyTo(img_background);
    return _img_foreground;
  }

  // difference naming style since I dont want to change the python API
  cv::cuda::GpuMat getBackgroundModel() { return img_background; }

  void init(const cv::cuda::GpuMat &img_input, cv::cuda::GpuMat &img_outfg,
            cv::cuda::GpuMat &img_outbg) {
    CV_Assert(img_input.empty() == false);
    img_outfg = cv::cuda::GpuMat(img_input.size(), CV_8UC1, cv::Scalar(0));
    img_outbg = cv::cuda::GpuMat(img_input.size(), CV_8UC3, cv::Scalar(0));
  }

  virtual void process(const cv::cuda::GpuMat &img_input,
                       cv::cuda::GpuMat &img_output,
                       cv::cuda::GpuMat &img_bgmodel) = 0;

  virtual std::ostream &dump(std::ostream &o) const {
    return o << get_algorithm_name();
  }
  std::string get_algorithm_name() const { return algorithm_name; }

private:
  friend std::ostream &operator<<(std::ostream &o,
                                  const std::shared_ptr<cuIBGS> &ibgs) {
    return ibgs.get()->dump(o);
  }
  friend std::ostream &operator<<(std::ostream &o, const cuIBGS *ibgs) {
    return ibgs->dump(o);
  }
  friend std::string to_string(const std::shared_ptr<cuIBGS> &ibgs) {
    std::ostringstream ss;
    ss << ibgs;
    return ss.str();
  }
};

class cuBGS_Factory {
public:
  cuBGS_Factory() {
    // debug_construction(cuBGS_Factory);
  }
  virtual ~cuBGS_Factory() {
    // debug_destruction(cuBGS_Factory);
  }
  static cuBGS_Factory *Instance() {
    static cuBGS_Factory factory;
    return &factory;
  }

  std::shared_ptr<cuIBGS> Create(std::string name) {
    cuIBGS *instance = nullptr;

    // find name in the registry and call factory method.
    auto it = factoryFunctionRegistry.find(name);
    if (it != factoryFunctionRegistry.end())
      instance = it->second();

    // wrap instance in a shared ptr and return
    if (instance != nullptr)
      return std::shared_ptr<cuIBGS>(instance);
    else
      return nullptr;
  }

  std::vector<std::string> GetRegisteredAlgorithmsName() {
    std::vector<std::string> algorithmsName;
    for (auto it = factoryFunctionRegistry.begin();
         it != factoryFunctionRegistry.end(); ++it) {
      algorithmsName.push_back(it->first);
    }
    return algorithmsName;
  }

  void
  RegisterFactoryFunction(std::string name,
                          std::function<cuIBGS *(void)> classFactoryFunction) {
    // register the class factory function
    factoryFunctionRegistry[name] = classFactoryFunction;
  }

private:
  std::map<std::string, std::function<cuIBGS *(void)>> factoryFunctionRegistry;
};

template <class T> class cuBGS_Register {
public:
  cuBGS_Register(const std::string className) {
    cuBGS_Factory::Instance()->RegisterFactoryFunction(
        className, [](void) -> cuIBGS * { return new T(); });
  }
  virtual ~cuBGS_Register() {}
};

} // namespace algorithms
} // namespace bgslibrary
