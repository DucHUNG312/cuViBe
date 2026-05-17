#include <exception>
#include <pybind11/pybind11.h>

#include <opencv2/opencv.hpp>

#include "../../bgslibrary/algorithms/algorithms.h"
#include "ndarray_converter.h"

#if CV_MAJOR_VERSION >= 4
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#endif

namespace py = pybind11;

cv::Mat transpose_image(const cv::Mat &image) {
  std::cerr << "Input size: " << image.size() << std::endl;
  std::cerr << "Returning the image transpose" << std::endl;
  return image.t();
}

void show_image(const cv::Mat &image) {
  cv::imshow("Image", image);
  cv::waitKey(0);
}

cv::Mat read_image(std::string image_name) {
  cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
  return image;
}

PYBIND11_MODULE(pybgs, m) {
  NDArrayConverter::init_numpy();
  m.doc() = "python wrapper for bgslibrary using pybind11";
  py::object version = py::cast("3.3.0");
  m.attr("__version__") = version;

  // Basic test
  m.def("read_image", &read_image, "A function that read an image",
        py::arg("image"));
  m.def("show_image", &show_image, "A function that show an image",
        py::arg("image"));
  m.def("transpose_image", &transpose_image,
        "A function that transpose an image", py::arg("image"));

  py::class_<FrameDifference>(m, "FrameDifference")
      .def(py::init<>())
      .def("apply", &FrameDifference::apply)
      .def("getBackgroundModel", &FrameDifference::getBackgroundModel);

  py::class_<StaticFrameDifference>(m, "StaticFrameDifference")
      .def(py::init<>())
      .def("apply", &StaticFrameDifference::apply)
      .def("getBackgroundModel", &StaticFrameDifference::getBackgroundModel);

  py::class_<WeightedMovingMean>(m, "WeightedMovingMean")
      .def(py::init<>())
      .def("apply", &WeightedMovingMean::apply)
      .def("getBackgroundModel", &WeightedMovingMean::getBackgroundModel);

  py::class_<WeightedMovingVariance>(m, "WeightedMovingVariance")
      .def(py::init<>())
      .def("apply", &WeightedMovingVariance::apply)
      .def("getBackgroundModel", &WeightedMovingVariance::getBackgroundModel);

  py::class_<AdaptiveBackgroundLearning>(m, "AdaptiveBackgroundLearning")
      .def(py::init<>())
      .def("apply", &AdaptiveBackgroundLearning::apply)
      .def("getBackgroundModel",
           &AdaptiveBackgroundLearning::getBackgroundModel);

  py::class_<AdaptiveSelectiveBackgroundLearning>(
      m, "AdaptiveSelectiveBackgroundLearning")
      .def(py::init<>())
      .def("apply", &AdaptiveSelectiveBackgroundLearning::apply)
      .def("getBackgroundModel",
           &AdaptiveSelectiveBackgroundLearning::getBackgroundModel);

  py::class_<MixtureOfGaussianV2>(m, "MixtureOfGaussianV2")
      .def(py::init<>())
      .def("apply", &MixtureOfGaussianV2::apply)
      .def("getBackgroundModel", &MixtureOfGaussianV2::getBackgroundModel);

#if CV_MAJOR_VERSION == 2
  py::class_<MixtureOfGaussianV1>(m, "MixtureOfGaussianV1")
      .def(py::init<>())
      .def("apply", &MixtureOfGaussianV1::apply)
      .def("getBackgroundModel", &MixtureOfGaussianV1::getBackgroundModel);
#endif

#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4 && CV_SUBMINOR_VERSION >= 3
  py::class_<GMG>(m, "GMG")
      .def(py::init<>())
      .def("apply", &GMG::apply)
      .def("getBackgroundModel", &GMG::getBackgroundModel);
#endif

#if CV_MAJOR_VERSION >= 3
  py::class_<KNN>(m, "KNN")
      .def(py::init<>())
      .def("apply", &KNN::apply)
      .def("getBackgroundModel", &KNN::getBackgroundModel);
#endif

#if CV_MAJOR_VERSION == 2 || CV_MAJOR_VERSION == 3
  py::class_<DPAdaptiveMedian>(m, "DPAdaptiveMedian")
      .def(py::init<>())
      .def("apply", &DPAdaptiveMedian::apply)
      .def("getBackgroundModel", &DPAdaptiveMedian::getBackgroundModel);

  py::class_<DPGrimsonGMM>(m, "DPGrimsonGMM")
      .def(py::init<>())
      .def("apply", &DPGrimsonGMM::apply)
      .def("getBackgroundModel", &DPGrimsonGMM::getBackgroundModel);

  py::class_<DPZivkovicAGMM>(m, "DPZivkovicAGMM")
      .def(py::init<>())
      .def("apply", &DPZivkovicAGMM::apply)
      .def("getBackgroundModel", &DPZivkovicAGMM::getBackgroundModel);

  py::class_<DPMean>(m, "DPMean")
      .def(py::init<>())
      .def("apply", &DPMean::apply)
      .def("getBackgroundModel", &DPMean::getBackgroundModel);

  py::class_<DPWrenGA>(m, "DPWrenGA")
      .def(py::init<>())
      .def("apply", &DPWrenGA::apply)
      .def("getBackgroundModel", &DPWrenGA::getBackgroundModel);

  py::class_<DPPratiMediod>(m, "DPPratiMediod")
      .def(py::init<>())
      .def("apply", &DPPratiMediod::apply)
      .def("getBackgroundModel", &DPPratiMediod::getBackgroundModel);

  py::class_<DPEigenbackground>(m, "DPEigenbackground")
      .def(py::init<>())
      .def("apply", &DPEigenbackground::apply)
      .def("getBackgroundModel", &DPEigenbackground::getBackgroundModel);

  py::class_<DPTexture>(m, "DPTexture")
      .def(py::init<>())
      .def("apply", &DPTexture::apply)
      .def("getBackgroundModel", &DPTexture::getBackgroundModel);

  py::class_<T2FGMM_UM>(m, "T2FGMM_UM")
      .def(py::init<>())
      .def("apply", &T2FGMM_UM::apply)
      .def("getBackgroundModel", &T2FGMM_UM::getBackgroundModel);

  py::class_<T2FGMM_UV>(m, "T2FGMM_UV")
      .def(py::init<>())
      .def("apply", &T2FGMM_UV::apply)
      .def("getBackgroundModel", &T2FGMM_UV::getBackgroundModel);

  py::class_<T2FMRF_UM>(m, "T2FMRF_UM")
      .def(py::init<>())
      .def("apply", &T2FMRF_UM::apply)
      .def("getBackgroundModel", &T2FMRF_UM::getBackgroundModel);

  py::class_<T2FMRF_UV>(m, "T2FMRF_UV")
      .def(py::init<>())
      .def("apply", &T2FMRF_UV::apply)
      .def("getBackgroundModel", &T2FMRF_UV::getBackgroundModel);

  py::class_<MultiCue>(m, "MultiCue")
      .def(py::init<>())
      .def("apply", &MultiCue::apply)
      .def("getBackgroundModel", &MultiCue::getBackgroundModel);
#endif

  py::class_<FuzzySugenoIntegral>(m, "FuzzySugenoIntegral")
      .def(py::init<>())
      .def("apply", &FuzzySugenoIntegral::apply)
      .def("getBackgroundModel", &FuzzySugenoIntegral::getBackgroundModel);

  py::class_<FuzzyChoquetIntegral>(m, "FuzzyChoquetIntegral")
      .def(py::init<>())
      .def("apply", &FuzzyChoquetIntegral::apply)
      .def("getBackgroundModel", &FuzzyChoquetIntegral::getBackgroundModel);

  py::class_<LBSimpleGaussian>(m, "LBSimpleGaussian")
      .def(py::init<>())
      .def("apply", &LBSimpleGaussian::apply)
      .def("getBackgroundModel", &LBSimpleGaussian::getBackgroundModel);

  py::class_<LBFuzzyGaussian>(m, "LBFuzzyGaussian")
      .def(py::init<>())
      .def("apply", &LBFuzzyGaussian::apply)
      .def("getBackgroundModel", &LBFuzzyGaussian::getBackgroundModel);

  py::class_<LBMixtureOfGaussians>(m, "LBMixtureOfGaussians")
      .def(py::init<>())
      .def("apply", &LBMixtureOfGaussians::apply)
      .def("getBackgroundModel", &LBMixtureOfGaussians::getBackgroundModel);

  py::class_<LBAdaptiveSOM>(m, "LBAdaptiveSOM")
      .def(py::init<>())
      .def("apply", &LBAdaptiveSOM::apply)
      .def("getBackgroundModel", &LBAdaptiveSOM::getBackgroundModel);

  py::class_<LBFuzzyAdaptiveSOM>(m, "LBFuzzyAdaptiveSOM")
      .def(py::init<>())
      .def("apply", &LBFuzzyAdaptiveSOM::apply)
      .def("getBackgroundModel", &LBFuzzyAdaptiveSOM::getBackgroundModel);

  py::class_<VuMeter>(m, "VuMeter")
      .def(py::init<>())
      .def("apply", &VuMeter::apply)
      .def("getBackgroundModel", &VuMeter::getBackgroundModel);

  py::class_<KDE>(m, "KDE")
      .def(py::init<>())
      .def("apply", &KDE::apply)
      .def("getBackgroundModel", &KDE::getBackgroundModel);

  py::class_<IndependentMultimodal>(m, "IndependentMultimodal")
      .def(py::init<>())
      .def("apply", &IndependentMultimodal::apply)
      .def("getBackgroundModel", &IndependentMultimodal::getBackgroundModel);

#if (CV_MAJOR_VERSION == 2) ||                                                 \
    (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 4 &&                         \
     CV_VERSION_REVISION <= 7)
  py::class_<LBP_MRF>(m, "LBP_MRF")
      .def(py::init<>())
      .def("apply", &LBP_MRF::apply)
      .def("getBackgroundModel", &LBP_MRF::getBackgroundModel);

  py::class_<MultiLayer>(m, "MultiLayer")
      .def(py::init<>())
      .def("apply", &MultiLayer::apply)
      .def("getBackgroundModel", &MultiLayer::getBackgroundModel);
#endif

  py::class_<PixelBasedAdaptiveSegmenter>(m, "PixelBasedAdaptiveSegmenter")
      .def(py::init<>())
      .def("apply", &PixelBasedAdaptiveSegmenter::apply)
      .def("getBackgroundModel",
           &PixelBasedAdaptiveSegmenter::getBackgroundModel);

  py::class_<SigmaDelta>(m, "SigmaDelta")
      .def(py::init<>())
      .def("apply", &SigmaDelta::apply)
      .def("getBackgroundModel", &SigmaDelta::getBackgroundModel);

  py::class_<SuBSENSE>(m, "SuBSENSE")
      .def(py::init<>())
      .def("apply", &SuBSENSE::apply)
      .def("getBackgroundModel", &SuBSENSE::getBackgroundModel);

  py::class_<LOBSTER>(m, "LOBSTER")
      .def(py::init<>())
      .def("apply", &LOBSTER::apply)
      .def("getBackgroundModel", &LOBSTER::getBackgroundModel);

  py::class_<PAWCS>(m, "PAWCS")
      .def(py::init<>())
      .def("apply", &PAWCS::apply)
      .def("getBackgroundModel", &PAWCS::getBackgroundModel);

  py::class_<TwoPoints>(m, "TwoPoints")
      .def(py::init<>())
      .def("apply", &TwoPoints::apply)
      .def("getBackgroundModel", &TwoPoints::getBackgroundModel);

  py::class_<ViBe>(m, "ViBe")
      .def(py::init<>())
      .def("apply", &ViBe::apply)
      .def("getBackgroundModel", &ViBe::getBackgroundModel);

  py::class_<CodeBook>(m, "CodeBook")
      .def(py::init<>())
      .def("apply", &CodeBook::apply)
      .def("getBackgroundModel", &CodeBook::getBackgroundModel);

  // CUDA submodule — pybgs.cu.ViBe
  // Input/output are CuPy ndarrays; data never leaves the GPU.
  py::module_ cu = m.def_submodule("cuda", "CUDA-accelerated algorithms");

  py::class_<cuViBe>(cu, "ViBe")
      .def(py::init<>())
      .def(
          "apply",
          [](cuViBe &self, py::object img) -> py::object {
            // Wrap CuPy input as GpuMat — zero-copy, CuPy owns the memory
            py::dict in_iface =
                img.attr("__cuda_array_interface__").cast<py::dict>();
            void *in_ptr = reinterpret_cast<void *>(
                in_iface["data"].cast<py::tuple>()[0].cast<uintptr_t>());
            auto shape = in_iface["shape"].cast<py::tuple>();
            int h = shape[0].cast<int>(), w = shape[1].cast<int>();
            int c = shape.size() >= 3 ? shape[2].cast<int>() : 1;
            size_t in_step = static_cast<size_t>(w * c);
            if (in_iface.contains("strides") &&
                !py::object(in_iface["strides"]).is_none()) {
              auto strides = in_iface["strides"].cast<py::tuple>();
              if (strides.size() > 0)
                in_step = strides[0].cast<size_t>();
            }
            cv::cuda::GpuMat gpu_in(h, w, c == 1 ? CV_8UC1 : CV_8UC3, in_ptr,
                                    in_step);

            cv::cuda::GpuMat gpu_out = self.apply(gpu_in);

            // Allocate CuPy output and copy into it via GpuMat::copyTo
            py::module_ cupy = py::module_::import("cupy");
            int out_c = gpu_out.channels();
            py::object out_arr =
                out_c > 1
                    ? cupy.attr("empty")(
                          py::make_tuple(gpu_out.rows, gpu_out.cols, out_c),
                          "uint8")
                    : cupy.attr("empty")(
                          py::make_tuple(gpu_out.rows, gpu_out.cols), "uint8");
            py::dict out_iface =
                out_arr.attr("__cuda_array_interface__").cast<py::dict>();
            void *out_ptr = reinterpret_cast<void *>(
                out_iface["data"].cast<py::tuple>()[0].cast<uintptr_t>());
            size_t row_bytes =
                static_cast<size_t>(gpu_out.cols) * gpu_out.elemSize();
            cv::cuda::GpuMat dst(gpu_out.rows, gpu_out.cols, gpu_out.type(),
                                 out_ptr, row_bytes);
            gpu_out.copyTo(dst);
            return out_arr;
          },
          py::arg("img"),
          "Apply background subtraction; img is a CuPy uint8 ndarray (H,W,3).")
      .def(
          "getBackgroundModel",
          [](cuViBe &self) -> py::object {
            cv::cuda::GpuMat gpu_bg = self.getBackgroundModel();
            if (gpu_bg.empty())
              return py::none();
            py::module_ cupy = py::module_::import("cupy");
            int c = gpu_bg.channels();
            py::object out_arr =
                c > 1
                    ? cupy.attr("empty")(
                          py::make_tuple(gpu_bg.rows, gpu_bg.cols, c), "uint8")
                    : cupy.attr("empty")(
                          py::make_tuple(gpu_bg.rows, gpu_bg.cols), "uint8");
            py::dict out_iface =
                out_arr.attr("__cuda_array_interface__").cast<py::dict>();
            void *out_ptr = reinterpret_cast<void *>(
                out_iface["data"].cast<py::tuple>()[0].cast<uintptr_t>());
            size_t row_bytes =
                static_cast<size_t>(gpu_bg.cols) * gpu_bg.elemSize();
            cv::cuda::GpuMat dst(gpu_bg.rows, gpu_bg.cols, gpu_bg.type(),
                                 out_ptr, row_bytes);
            gpu_bg.copyTo(dst);
            return out_arr;
          },
          "Returns background model as a CuPy ndarray, or None if not yet "
          "initialized.");
}
