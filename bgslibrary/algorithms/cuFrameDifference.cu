#include "cuFrameDifference.h"

namespace bgslibrary {
namespace algorithms {

namespace {

constexpr int NUM_THREADS = 256;

__global__ void cuFrameDifference_kernel_8u_c3r(
    const uint8_t *__restrict__ img_input, const uint8_t *__restrict__ bg_input,
    uint8_t *__restrict__ img_output, int width, int height, int in_step,
    int bg_step, int out_step, float threshold) {
  const auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx >= width * height) {
    return;
  }

  const int x = tidx % width;
  const int y = tidx / width;

  float gray = 0.299f * abs(img_input[y * in_step + 3 * x] -
                            bg_input[y * bg_step + 3 * x]) +
               0.587f * abs(img_input[y * in_step + 3 * x + 1] -
                            bg_input[y * bg_step + 3 * x + 1]) +
               0.114f * abs(img_input[y * in_step + 3 * x + 2] -
                            bg_input[y * bg_step + 3 * x + 2]);
  img_output[y * out_step + x] = (gray > threshold) ? 255 : 0;
}

void cuFrameDifference_8u_c3r(const cv::cuda::GpuMat &img_input,
                              const cv::cuda::GpuMat &bg_input,
                              cv::cuda::GpuMat &img_output, float threshold) {
  CV_Assert(img_input.size() == bg_input.size());
  CV_Assert(img_input.size() == img_output.size());
  const int width = static_cast<int>(img_input.cols);
  const int height = static_cast<int>(img_input.rows);
  const int blocks = (width * height + NUM_THREADS - 1) / NUM_THREADS;

  cuFrameDifference_kernel_8u_c3r<<<blocks, NUM_THREADS>>>(
      img_input.data, bg_input.data, img_output.data, width, height,
      static_cast<int>(img_input.step), static_cast<int>(bg_input.step),
      static_cast<int>(img_output.step), threshold);

  CUDA_Check(cudaGetLastError());
}

} // namespace

cuFrameDifference::cuFrameDifference()
    : cuIBGS(quote(cuFrameDifference)), threshold(15) {
  debug_construction(cuFrameDifference);
  initLoadSaveConfig(algorithm_name);
}

cuFrameDifference::~cuFrameDifference() {
  debug_destruction(cuFrameDifference);
}

void cuFrameDifference::process(const cv::cuda::GpuMat &img_input,
                                cv::cuda::GpuMat &img_output) {
  if (img_foreground.empty()) {
    img_foreground = cv::cuda::GpuMat(img_input.size(), CV_8UC1, cv::Scalar(0));
  }
  if (img_background.empty()) {
    img_input.copyTo(img_background);
    first_time = false;
    return;
  }

  cuFrameDifference_8u_c3r(img_background, img_input, img_foreground,
                           static_cast<float>(threshold));

  img_foreground.copyTo(img_output);
  img_input.copyTo(img_background);
}

void cuFrameDifference::save_config(cv::FileStorage &fs) {
  fs << "threshold" << threshold;
}

void cuFrameDifference::load_config(cv::FileStorage &fs) {
  fs["threshold"] >> threshold;
}

} // namespace algorithms

} // namespace bgslibrary
