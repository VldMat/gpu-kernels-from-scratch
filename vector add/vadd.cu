#include <cstdio>
#include <cmath>
#include <cstdlib>


inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d\n",
                cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void vadd(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main (){
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a,bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n+threads - 1)/ threads;
    vadd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());     //check errors
    CUDA_CHECK(cudaDeviceSynchronize());   //check synchronization

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost)); //copy results

    double max_error = 0.0;
    for(int i = 0; i < n; i++) max_error = fmax(max_error, fabs(h_c[i] - 3.0f));
    printf("maxerror: %f(expect 0.0)\n", max_error);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;

}