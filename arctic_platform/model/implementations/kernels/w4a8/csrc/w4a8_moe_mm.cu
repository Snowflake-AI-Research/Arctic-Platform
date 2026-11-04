// Grouped INT4 x FP8 GEMM for Hopper (SM90a), ported from SGLang's sgl-kernel
// (csrc/moe/cutlass_moe/w4a8, Apache-2.0) for training: tile configurations chosen by the caller.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<S == Sched::PP, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
                                            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule = std::conditional_t<S == Sched::PP, cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
                                              cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

#define W4A8_ARGS d, a, b, a_scales, b_scales, expert_offsets, problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size

}  // namespace

// Tiles with a 128-wide K step (one INT4 scale group per step, scales in the [E, groups, out] layout), sized for
// training's thousands of rows per expert. Measured on H200 for GLM-5.3 at EP8 (32 experts, ~8.5K rows each):
// 128x192 runs 899 / 829 TFLOP/s on gate / down against 641 / 610 for decode + BF16 torch._grouped_mm, where
// SGLang's 128x64x512 serving tile runs 505 / 470.
// config: 0 = 128x192 cooperative, 1 = 128x128 cooperative, 4 = 128x192 cooperative with a 2x1 cluster.
// (128x160 and 128x224 compile but return wrong results: 0.9 relative error; 2 and 3 are left unassigned.)
void w4a8_moe_mm(torch::Tensor d, torch::Tensor a, torch::Tensor b, torch::Tensor a_scales, torch::Tensor b_scales,
                 torch::Tensor expert_offsets, torch::Tensor problem_sizes, torch::Tensor a_strides,
                 torch::Tensor b_strides, torch::Tensor d_strides, torch::Tensor s_strides, int64_t chunk_size,
                 int64_t config) {
  const c10::cuda::CUDAGuard device_guard(a.device());
  switch (config) {
    case 0: cutlass_w4a8_group_gemm_caller<SM90W4A8Config<128, 192, 128, 1, 1, 1, Sched::CO>::Gemm>(W4A8_ARGS); break;
    case 1: cutlass_w4a8_group_gemm_caller<SM90W4A8Config<128, 128, 128, 1, 1, 1, Sched::CO>::Gemm>(W4A8_ARGS); break;
    case 4: cutlass_w4a8_group_gemm_caller<SM90W4A8Config<128, 192, 128, 2, 1, 1, Sched::CO>::Gemm>(W4A8_ARGS); break;
    default: TORCH_CHECK(false, "unknown w4a8 tile config ", config);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("w4a8_moe_mm", &w4a8_moe_mm, "grouped INT4 x FP8 GEMM (SM90a)"); }
