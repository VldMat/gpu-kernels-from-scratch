#include <cstdlib>
#include <cstdio>
#include <cmath>


inline void cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void relu(const float* x, float* out, int n){
    int i = blockIdx.x * blockDim.x +threadIdx.x;
    if (i < n) out[i] = max(x[i], 0.0f);
}

int main(){
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    
    float *h_x = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    
    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    for (int i = 0; i < n; i++) {h_x[i] = float(i % 7) - 3.0f;}

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu<<<blocks, threads>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int i = 0; i < n; i++){
        float expected = h_x[i]> 0.0f ? h_x[i] : 0.0f;
        max_error = fmax(max_error,fabs(h_out[i] - expected));
    }
    printf("maxerror: %f (expect 0.0)\n", max_error);

    cudaFree(d_x); cudaFree(d_out);
    free(h_x); free(h_out);
    return 0;
}