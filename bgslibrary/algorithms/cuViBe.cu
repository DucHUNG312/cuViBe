#include "cuViBe.h"
#include <cstdlib>
#include <memory>
#include <vector>

#define CUDA_Check(func)                                                       \
  {                                                                            \
    cudaError_t ret = func;                                                    \
    if (ret != cudaSuccess) {                                                  \
      CV_Assert(false);                                                        \
    }                                                                          \
  }

namespace bgslibrary {
namespace algorithms {
namespace {

constexpr int NUM_THREADS = 256;
constexpr int COLOR_BACKGROUND = 0;
constexpr int COLOR_FOREGROUND = 255;
constexpr int NUMBER_OF_HISTORY_IMAGES = 2;
constexpr int DEFAULT_NUM_SAMPLES = 20;
constexpr int DEFAULT_MATCH_THRESH = 20;
constexpr int DEFAULT_MATCH_NUM = 2;
constexpr int DEFAULT_UPDATE_FACTOR = 16;

__forceinline__ __host__ __device__ uint32_t xorshift32(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

__forceinline__ __host__ __device__ uint32_t
distance_Han2014Improved(uint8_t pixel, uint8_t bg) {
  uint8_t min, max;

  // Computes R = 0.13 min{ max[bg,26], 230}
  max = 26;
  if (bg > max) {
    max = bg;
  }

  min = 230;
  if (min > max) {
    min = max;
  }

  return (uint32_t)(0.13 * min);
}

__forceinline__ __host__ __device__ int abs_uint(const int i) {
  return (i >= 0) ? i : -i;
}

__forceinline__ __host__ __device__ int32_t
distance_is_close_8u_c3r(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2,
                         uint8_t g2, uint8_t b2, uint32_t threshold) {
  return (abs_uint(r1 - r2) + abs_uint(g1 - g2) + abs_uint(b1 - b2) <=
          4.5 * threshold);
}

// Initialise history_buffer on-device: each pixel gets num_extra samples seeded
// from its 3-channel value in the first frame +/- uniform noise in [-10, +10].
template <const int NUM_EXTRA>
__global__ static void cuViBeModel_init_history_buffer_kernel_8u_c3r(
    uint8_t *__restrict__ history_buffer, const uint8_t *__restrict__ image,
    size_t image_step, int width, int height, uint32_t seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height) {
    return;
  }

  const int x = idx % width;
  const int y = idx / width;
  const uint8_t ch[3] = {image[y * image_step + 3 * x],
                         image[y * image_step + 3 * x + 1],
                         image[y * image_step + 3 * x + 2]};

  uint32_t state = seed ^ (static_cast<uint32_t>(idx) * 2654435761u);
#pragma unroll
  for (int s = 0; s < NUM_EXTRA; ++s) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      int noise = static_cast<int>(xorshift32(state) % 20u) - 10;
      int v = static_cast<int>(ch[c]) + noise;
      history_buffer[idx * (NUM_EXTRA * 3) + s * 3 + c] =
          static_cast<uint8_t>(min(255, max(0, v)));
    }
  }
}

template <int NUM_EXTRA>
__global__ static void cuViBeModel_segmentation_update_kernel_8u_c3r(
    const uint8_t *__restrict__ image_data, size_t image_step,
    uint8_t *history_image, uint8_t *__restrict__ history_buffer,
    uint8_t *__restrict__ seg_map, size_t seg_step, int width, int height,
    int matching_number, uint32_t matching_threshold, int swapping_slot,
    uint32_t *__restrict__ jump, int32_t *__restrict__ neighbor,
    uint32_t *__restrict__ position, uint32_t global_seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height) {
    return;
  }

  const int x = idx % width;
  const int y = idx / width;
  const uint8_t r = image_data[y * image_step + 3 * x];
  const uint8_t g = image_data[y * image_step + 3 * x + 1];
  const uint8_t b = image_data[y * image_step + 3 * x + 2];

  // ---- Segmentation ----
  int count = matching_number - 1;

  const uint8_t *first = history_image;
  if (!distance_is_close_8u_c3r(r, g, b, first[3 * idx], first[3 * idx + 1],
                                first[3 * idx + 2], matching_threshold)) {
    count = matching_number;
  }

#pragma unroll
  for (int i = 1; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
    const uint8_t *pels = history_image + i * 3 * width * height;
    if (distance_is_close_8u_c3r(r, g, b, pels[3 * idx], pels[3 * idx + 1],
                                 pels[3 * idx + 2], matching_threshold)) {
      --count;
    }
  }

  if (count > 0) {
    uint8_t *swap_buf = history_image + swapping_slot * 3 * width * height;
    const int buf_base = (3 * idx) * NUM_EXTRA;

    for (int i = 0; i < NUM_EXTRA; ++i) {
      const int buf_idx = buf_base + i * 3;

      if (distance_is_close_8u_c3r(
              r, g, b, history_buffer[buf_idx], history_buffer[buf_idx + 1],
              history_buffer[buf_idx + 2], matching_threshold)) {
        --count;
      }

      uint8_t tr = swap_buf[3 * idx], tg = swap_buf[3 * idx + 1],
              tb = swap_buf[3 * idx + 2];
      swap_buf[3 * idx] = history_buffer[buf_idx];
      swap_buf[3 * idx + 1] = history_buffer[buf_idx + 1];
      swap_buf[3 * idx + 2] = history_buffer[buf_idx + 2];
      history_buffer[buf_idx] = tr;
      history_buffer[buf_idx + 1] = tg;
      history_buffer[buf_idx + 2] = tb;

      if (count <= 0) {
        break;
      }
    }
  }

  const bool is_fg = (count > 0);
  seg_map[y * seg_step + x] = is_fg ? COLOR_FOREGROUND : COLOR_BACKGROUND;

  // ---- Update (background pixels only) ----
  if (is_fg) {
    return;
  }

  uint32_t state = global_seed ^ (static_cast<uint32_t>(idx + 1) * 2654435761u);
  const int arr_size = (width > height) ? 2 * width + 1 : 2 * height + 1;
  const int shift = xorshift32(state);
  const int arr_idx = static_cast<int>(shift % static_cast<uint32_t>(arr_size));
  if (shift % jump[arr_idx] != 0u) {
    return;
  }

  const int pos = static_cast<int>(position[arr_idx]);
  const int nbr_off = neighbor[arr_idx];

#define WRITE_SAMPLE(pixel_idx)                                                \
  if (pos < NUMBER_OF_HISTORY_IMAGES) {                                        \
    int _off = 3 * (pixel_idx) + pos * 3 * width * height;                     \
    history_image[_off] = r;                                                   \
    history_image[_off + 1] = g;                                               \
    history_image[_off + 2] = b;                                               \
  } else {                                                                     \
    int _off =                                                                 \
        (3 * (pixel_idx)) * NUM_EXTRA + 3 * (pos - NUMBER_OF_HISTORY_IMAGES);  \
    history_buffer[_off] = r;                                                  \
    history_buffer[_off + 1] = g;                                              \
    history_buffer[_off + 2] = b;                                              \
  }

  WRITE_SAMPLE(idx);

  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    WRITE_SAMPLE(idx + nbr_off);
  }

#undef WRITE_SAMPLE
}

void cuViBeModel_init_8u_c3r(cuViBe::cuViBeModel *model,
                             const cv::cuda::GpuMat &img_input) {
  CV_Assert(model != nullptr);
  CV_Assert(img_input.data != nullptr);
  CV_Assert(img_input.cols > 0 && img_input.rows > 0);
  CV_Assert(img_input.channels() == 3);

  const uint32_t width = static_cast<uint32_t>(img_input.cols);
  const uint32_t height = static_cast<uint32_t>(img_input.rows);
  model->width = width;
  model->height = height;
  model->number_of_samples = DEFAULT_NUM_SAMPLES;

  {
    const size_t row_bytes = 3 * width * sizeof(uint8_t);
    CUDA_Check(cudaMalloc(&model->history_image, NUMBER_OF_HISTORY_IMAGES * 3 *
                                                     width * height *
                                                     sizeof(uint8_t)));
    for (int i = 0; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
      CUDA_Check(cudaMemcpy2D(model->history_image + i * 3 * width * height,
                              row_bytes, img_input.data, img_input.step,
                              row_bytes, height, cudaMemcpyDeviceToDevice));
    }
  }

  {
    constexpr uint32_t NUM_EXTRA =
        DEFAULT_NUM_SAMPLES - NUMBER_OF_HISTORY_IMAGES;
    CUDA_Check(cudaMalloc(&model->history_buffer,
                          3 * width * height * NUM_EXTRA * sizeof(uint8_t)));
    const int blocks =
        (static_cast<int>(width * height) + NUM_THREADS - 1) / NUM_THREADS;
    cuViBeModel_init_history_buffer_kernel_8u_c3r<NUM_EXTRA>
        <<<blocks, NUM_THREADS>>>(model->history_buffer, img_input.data,
                                  img_input.step, static_cast<int>(width),
                                  static_cast<int>(height),
                                  static_cast<uint32_t>(rand()));
    CUDA_Check(cudaGetLastError());
  }

  {
    const int size =
        static_cast<int>((width > height) ? 2 * width + 1 : 2 * height + 1);
    std::vector<uint32_t> h_jump(size);
    std::vector<int32_t> h_neighbor(size);
    std::vector<uint32_t> h_position(size);
    for (int i = 0; i < size; ++i) {
      h_jump[i] =
          static_cast<uint32_t>((rand() % (2 * model->update_factor)) + 1);
      h_neighbor[i] = static_cast<int32_t>(
          ((rand() % 3) - 1) + ((rand() % 3) - 1) * static_cast<int>(width));
      h_position[i] = static_cast<uint32_t>(rand() % model->number_of_samples);
    }
    CUDA_Check(cudaMalloc(&model->jump, size * sizeof(uint32_t)));
    CUDA_Check(cudaMalloc(&model->neighbor, size * sizeof(int32_t)));
    CUDA_Check(cudaMalloc(&model->position, size * sizeof(uint32_t)));
    CUDA_Check(cudaMemcpy(model->jump, h_jump.data(), size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_Check(cudaMemcpy(model->neighbor, h_neighbor.data(),
                          size * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_Check(cudaMemcpy(model->position, h_position.data(),
                          size * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }
}

void cuViBeModel_segmentation_update_8u_c3r(cuViBe::cuViBeModel *model,
                                            const cv::cuda::GpuMat &img_input,
                                            cv::cuda::GpuMat &seg_map) {
  CV_Assert(model != nullptr);
  CV_Assert(model->history_buffer != nullptr);
  CV_Assert(model->jump != nullptr && model->neighbor != nullptr &&
            model->position != nullptr);

  const int width = static_cast<int>(model->width);
  const int height = static_cast<int>(model->height);

  model->last_history_image_swapped =
      (model->last_history_image_swapped + 1) % NUMBER_OF_HISTORY_IMAGES;
  const int swapping_slot = static_cast<int>(model->last_history_image_swapped);

  constexpr int NUM_EXTRA = DEFAULT_NUM_SAMPLES - NUMBER_OF_HISTORY_IMAGES;
  const int blocks = (width * height + NUM_THREADS - 1) / NUM_THREADS;

  cuViBeModel_segmentation_update_kernel_8u_c3r<NUM_EXTRA>
      <<<blocks, NUM_THREADS>>>(
          img_input.data, img_input.step, model->history_image,
          model->history_buffer, seg_map.data, seg_map.step, width, height,
          static_cast<int>(model->matching_number), model->matching_threshold,
          swapping_slot, model->jump, model->neighbor, model->position,
          static_cast<uint32_t>(rand()));

  CUDA_Check(cudaGetLastError());
}

void cuViBeModel_free(cuViBe::cuViBeModel *model) {
  if (model == nullptr) {
    return;
  }
  CUDA_Check(cudaFree(model->history_image));
  CUDA_Check(cudaFree(model->history_buffer));
  CUDA_Check(cudaFree(model->jump));
  CUDA_Check(cudaFree(model->neighbor));
  CUDA_Check(cudaFree(model->position));
  delete model;
}

void cuViBeModel_set_matching_threshold(
    cuViBe::cuViBeModel *model,
    const std::optional<uint32_t> &matching_threshold) {
  CV_Assert(model != nullptr);
  model->matching_threshold = matching_threshold.has_value()
                                  ? matching_threshold.value()
                                  : DEFAULT_MATCH_THRESH;
}

void cuViBeModel_set_matching_number(
    cuViBe::cuViBeModel *model,
    const std::optional<uint32_t> &matching_number) {
  CV_Assert(model != nullptr);
  model->matching_number =
      matching_number.has_value() ? matching_number.value() : DEFAULT_MATCH_NUM;
}

void cuViBeModel_set_update_factor(
    cuViBe::cuViBeModel *model, const std::optional<uint32_t> &update_factor) {
  CV_Assert(model != nullptr);
  const uint32_t factor =
      update_factor.has_value() ? update_factor.value() : DEFAULT_UPDATE_FACTOR;
  CV_Assert(factor > 0);
  model->update_factor = factor;

  if (model->jump != nullptr) {
    const int size = static_cast<int>((model->width > model->height)
                                          ? 2 * model->width + 1
                                          : 2 * model->height + 1);
    std::vector<uint32_t> h_jump(size);
    for (int i = 0; i < size; ++i) {
      h_jump[i] = (factor == 1)
                      ? 1u
                      : static_cast<uint32_t>(rand() % (2 * factor)) + 1u;
    }
    CUDA_Check(cudaMemcpy(model->jump, h_jump.data(), size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
  }
}

} // namespace

cuViBe::cuViBe() : model(new cuViBeModel(), &cuViBeModel_free) {}

cuViBe::~cuViBe() {}

void cuViBe::process(const cv::cuda::GpuMat &img_input,
                     cv::cuda::GpuMat &img_output,
                     cv::cuda::GpuMat &img_bgmodel) {
  init(img_input, img_output, img_bgmodel);

  if (img_input.empty()) {
    return;
  }

  cuViBeModel *model_ptr = model.get();
  if (this->first_time) {
    cuViBeModel_set_matching_threshold(model_ptr, matching_threshold);
    cuViBeModel_set_matching_number(model_ptr, matching_number);
    cuViBeModel_set_update_factor(model_ptr, update_factor);
    cuViBeModel_init_8u_c3r(model_ptr, img_input);
    this->first_time = false;
  }

  cuViBeModel_segmentation_update_8u_c3r(model_ptr, img_input, img_output);
}

void cuViBe::save_config(cv::FileStorage &fs) {
  if (matching_threshold.has_value()) {
    fs << "matching_threshold" << static_cast<int>(matching_threshold.value());
  }
  if (matching_number.has_value()) {
    fs << "matching_number" << static_cast<int>(matching_number.value());
  }
  if (update_factor.has_value()) {
    fs << "update_factor" << static_cast<int>(update_factor.value());
  }
}

void cuViBe::load_config(cv::FileStorage &fs) {
  int val;
  if (!fs["matching_threshold"].empty()) {
    fs["matching_threshold"] >> val;
    matching_threshold = static_cast<uint32_t>(val);
  }
  if (!fs["matching_number"].empty()) {
    fs["matching_number"] >> val;
    matching_number = static_cast<uint32_t>(val);
  }
  if (!fs["update_factor"].empty()) {
    fs["update_factor"] >> val;
    update_factor = static_cast<uint32_t>(val);
  }
}

} // namespace algorithms
} // namespace bgslibrary