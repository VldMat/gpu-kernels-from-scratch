#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#define BM 64            // block computes a 64x64 tile of C
#define BN 64
#define BK 8            // K is marched in chunks of 8
#define TM 8           // each thread computes 8 outputs (a vertical strip)



inline void cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void matmul_reg(const float* A, const float* B, float* C, int N){
    int cRow = blockIdx.y;          // which 64-row block of C
    int cCol = blockIdx.x;          // which 64-col block of C


    __shared__ float As[BM*BK];     // 64 x 8 tile of A
    __shared__ float Bs[BK*BN];     // 8 x 64 tile of B

    int threadCol = threadIdx.x % BN;    // 0..63  -> this thread's column in the block
    int threadRow = threadIdx.x / BN;    // 0..7   -> which strip of 8 rows

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < N; bkIdx += BK){
        //load one element of each tile from the global -> shared
        As[innerRowA*BK+innerColA] = A[(cRow*BM + innerRowA)*N + (bkIdx + innerColA)];
        Bs[innerRowB*BN+innerColB] = B[(bkIdx +innerRowB)*N + (cCol*BN + innerColB)];
        __syncthreads();

        for (int dotIdx = 0; dotIdx< BK; dotIdx ++){
            float tmpB = Bs[dotIdx*BN + threadCol];
            for (int resIdx = 0; resIdx < TM; resIdx ++){
                threadResults[resIdx] += As[(threadRow*TM +resIdx)* BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for(int resIdx = 0; resIdx < TM; resIdx++){
        C[(cRow*BM + threadRow*TM + resIdx)*N + (cCol*BN + threadCol)] = threadResults[resIdx];
    }


}

int main(){
    int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }  // all ones

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // TODO 2: launch config — dim3 threads(16,16); dim3 blocks(...); then launch matmul<<<...>>>
    dim3 threads(BM * BN / TM);      // = 512
    dim3 blocks(N / BN, N / BM);     // = (16, 16) for N=1024
    matmul_reg<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    matmul_reg<<<blocks, threads>>>(d_A, d_B, d_C, N);   // warm-up run (first launch has extra overhead)
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matmul_reg<<<blocks, threads>>>(d_A, d_B, d_C, N);   // the timed run
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
    printf("time: %.3f ms   %.1f GFLOP/s\n", ms, gflops);

    // ---- cuBLAS benchmark ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // warm-up: cuBLAS loads and autotunes its kernel on the first call
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_cublas = 0.0f;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    double gflops_cublas = (2.0 * N * N * N) / (ms_cublas / 1000.0) / 1e9;
    printf("cuBLAS: %.3f ms   %.1f GFLOP/s\n", ms_cublas, gflops_cublas);

    // verify cuBLAS result (it wrote into d_C)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    double err_cublas = 0.0;
    for (int i = 0; i < N*N; i++) err_cublas = fmax(err_cublas, fabs(h_C[i] - (float)N));
    printf("cuBLAS max error: %f\n", err_cublas);

    cublasDestroy(handle);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // With A and B all ones, every C element = sum of N ones = N.
    double max_error = 0.0;
    for (int i = 0; i < N*N; i++) max_error = fmax(max_error, fabs(h_C[i] - (float)N));
    printf("max error: %f  (expect 0.0, every element should be %d)\n", max_error, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}