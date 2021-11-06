#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

// #define MAX(a, b) ((a < b) ? (b) : (a))
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
using namespace std;

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void fused_forward_tb_kernel(
    int m, int nnz, int h, int f,
    const float *attn_row, const float *attn_col,
    const int *row_ptr, const int *col_ind,
    const float *in_feat,
    const float negative_slope,
    const int *tile_scheduler,
    float *edge_max, float *edge_sum,
    float *out_feat)
{
    int tile_id = blockIdx.x;
    int rid = tile_scheduler[tile_id];
    int hid = blockIdx.y;
    int lb = row_ptr[rid];
    int hb = row_ptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int rh_id = rid * h + hid;
    int fid = blockIdx.y;

    if (ptr < hb)
    {
        int cid = col_ind[ptr];
        float row_val = attn_row[rh_id];
        float edge_val = row_val + attn_col[cid * h + hid];
        edge_val = LeakyRelu(edge_val, negative_slope);

        float weightMax = edge_val;
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            float tmp = __shfl_xor_sync(0xffffffff, weightMax, stride, 32);
            weightMax = MAX(tmp, weightMax);
        }
        if (threadIdx.x == 0)
            atomicMaxFloat(&edge_max[rh_id], weightMax);
        weightMax = edge_max[rh_id];

        edge_val = exp(edge_val - weightMax);

        float expAll = edge_val;
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            float tmp = __shfl_xor_sync(0xffffffff, expAll, stride, 32);
            expAll += tmp;
        }
        if (threadIdx.x == 0)
            atomicAdd(&edge_sum[rh_id], expAll);
        expAll = edge_max[rh_id];

        edge_val = edge_val / expAll;

        float partial_sum = 0;
        for (int stride = 16; stride > 0; stride >>= 1)
        {
            float product = in_feat[cid * h * f + hid * f + fid] * edge_val;
            float tmp = __shfl_xor_sync(0xffffffff, product, stride, 32);
            partial_sum += tmp;
        }
        if (threadIdx.x == 0)
            atomicAdd(&out_feat[rid * h * f + hid * f + fid], partial_sum);
    }
}

std::vector<torch::Tensor>
gat_tb_forward_cuda(
    torch::Tensor attn_row, torch::Tensor attn_col,
    torch::Tensor row_ptr, torch::Tensor col_ind,
    float negative_slope, torch::Tensor in_feat,
    torch::Tensor tile_scheduler)
{
    const auto m = row_ptr.size(0) - 1;
    const auto nnz = col_ind.size(0);
    const auto h = attn_row.size(1);
    const auto f = in_feat.size(2);
    const auto tile_num = tile_scheduler.size(0);
    auto devid = attn_row.device().index();
    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out_feat = torch::empty({m, h, f}, options);
    // auto edge_relu_csr = torch::empty({nnz, h}, options);
    // auto edge_softmax_csr = torch::empty({nnz, h}, options);
    auto edge_max = torch::empty({m, h}, options);
    auto edge_sum = torch::empty({m, h}, options);
    fused_forward_tb_kernel<<<dim3(tile_num, f, 1), dim3(32, h, 1)>>>(
        m, nnz, h, f,
        attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
        row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(),
        in_feat.data_ptr<float>(),
        negative_slope,
        tile_scheuler.data_ptr<int>(),
        edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(),
        out_feat.data_ptr<float>());
    return {out_feat, edge_max, edge_sum};
}