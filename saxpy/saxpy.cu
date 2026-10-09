#include <cstdio>
#include <cmath>
#include <cstdlib>

inline void cudaCheck(cudaError_t err,const char* file, int line){
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", 
            cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void saxpy(const float* x, float* y, float a, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <n) y[i] = a * x[i] + y[i];

}

int main(){
    int n = 1 << 20; float a = 2.0f;
    size_t bytes = n * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x,bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    for (int i = 0; i < n; i++) {h_x[i] = 1.0f; h_y[i] = 1.0f;}

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n+threads -1)/threads;
    saxpy<<<blocks, threads>>>(d_x, d_y, a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int i = 0; i < n; i++) max_error = fmax(max_error, fabs(h_y[i] - 3.0f));
    printf("maxerror: %f(expect 0.0)\n", max_error);
    
    cudaFree(d_x); cudaFree(d_y); 
    free(h_x); free(h_y);
    return 0;
}