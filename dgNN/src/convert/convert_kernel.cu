#include "../util/computeUtil.h"
#include <cuda.h>
#include <cusparse.h>
#include <torch/types.h>

void csr2cscKernel(int m, int n, int nnz, int *csrRowPtr, int *csrColInd,
                   float *csrVal, int *cscColPtr, int *cscRowInd,
                   float *cscVal)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle));
    size_t bufferSize = 0;
    void *buffer = NULL;
    checkCuSparseError(cusparseCsr2cscEx2_bufferSize(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
        cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &bufferSize));
    checkCudaError(cudaMalloc((void **)&buffer, bufferSize * sizeof(float)));
    checkCuSparseError(cusparseCsr2cscEx2(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
        cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer));
    checkCudaError(cudaFree(buffer));
}

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal)
{
    const auto n = csrRowPtr.size(0) - 1;
    const auto nnz = csrColInd.size(0);
    auto devid = csrRowPtr.device().index();
    auto optionsF =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto optionsI =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto cscColPtr = torch::empty({n + 1}, optionsI);
    auto cscRowInd = torch::empty({nnz}, optionsI);
    auto cscVal = torch::empty({nnz}, optionsF);
    csr2cscKernel(n, n, nnz, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(),
                  csrVal.data_ptr<float>(), cscColPtr.data_ptr<int>(),
                  cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>());
    return {cscColPtr, cscRowInd, cscVal};
}

void coo2csrKernel(int m, int nnz, int *cooRowInd, int *csrRowPtr)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle));
    checkCuSparseError(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
}

torch::Tensor coo2csr_cuda(torch::Tensor cooRowInd, int m)
{
    const auto nnz = cooRowInd.size(0);
    auto devid = cooRowInd.device().index();
    auto optionsI = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto csrRowPtr = torch::empty({m + 1}, optionsI);
    coo2csrKernel(m, nnz, cooRowInd.data_ptr<int>(), csrRowPtr.data_ptr<int>());
    return csrRowPtr;
}