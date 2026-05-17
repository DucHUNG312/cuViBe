#pragma once

#include "cuIBGS.h"

namespace bgslibrary {
namespace algorithms {

// CUDA version of ViBe
class cuViBe : public cuIBGS {
public:
  struct cuViBeModel {
    uint32_t width;
    uint32_t height;
    uint32_t number_of_samples;
    uint32_t matching_threshold;
    uint32_t matching_number;
    uint32_t update_factor;

    uint8_t *history_image = nullptr;
    uint8_t *history_buffer = nullptr;
    uint32_t last_history_image_swapped = 0;

    uint32_t *jump = nullptr;
    int32_t *neighbor = nullptr;
    uint32_t *position = nullptr;
  };

private:
  std::optional<uint32_t> matching_threshold;
  std::optional<uint32_t> matching_number;
  std::optional<uint32_t> update_factor;
  std::unique_ptr<cuViBeModel, void (*)(cuViBeModel *)> model;

public:
  cuViBe();
  ~cuViBe();

  void process(const cv::cuda::GpuMat &img_input, cv::cuda::GpuMat &img_output,
               cv::cuda::GpuMat &img_bgmodel) override;

private:
  void save_config(cv::FileStorage &fs);
  void load_config(cv::FileStorage &fs);
};

cubgs_register(cuViBe);

} // namespace algorithms
} // namespace bgslibrary
