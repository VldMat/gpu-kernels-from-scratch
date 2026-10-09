#include <cstdio>
#include <cmath>
#include <cstdlib>

inline void cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)


 __global__ void matmul(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) sum += A[row*N + k] * B[k * N + col];
    C[row * N +col] = sum;
 
 }

int main(){
    int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }  

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    
    dim3 threads(16, 16);
    dim3 blocks ((N+15)/16, (N+15)/16);
    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);   
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
    printf("time: %.3f ms   %.1f GFLOP/s\n", ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    
    double max_error = 0.0;
    for (int i = 0; i < N*N; i++) max_error = fmax(max_error, fabs(h_C[i] - (float)N));
    printf("max error: %f  (expect 0.0, every element should be %d)\n", max_error, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}