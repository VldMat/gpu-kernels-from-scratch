#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK 256





inline void cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void reduce(const float* in, float* out, int n){
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if ( tid < s) sdata[tid] += sdata[tid +s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];

    __syncthreads();
}

int main(){
    int n = 1 << 20;
    int blocks = (n + BLOCK - 1)/BLOCK;
    size_t in_bytes = (size_t)n *sizeof(float);
    size_t out_bytes = (size_t)blocks *sizeof(float);

    float *h_in = (float*)malloc(in_bytes);
    float *h_out = (float*)malloc(out_bytes);
    for (int i = 0; i < n; i++) h_in[i] = 3.0f;


    float  *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    


    int threads = BLOCK;
    reduce<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    reduce<<<blocks, threads>>>(d_in, d_out,  n);   // warm-up run (first launch has extra overhead)
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    reduce<<<blocks, threads>>>(d_in, d_out, n);   // the timed run
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gbps = (double)in_bytes / (ms / 1000.0) / 1e9;
    printf("time: %.3f ms   %.1f GB/s\n", ms, gbps);

    CUDA_CHECK(cudaGetLastError());



    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (int b = 0; b < blocks; b++) total += h_out[b];   // sum the partials
    double expected = 3.0 * (double)n;
    printf("total: %f  expected: %f  error: %f\n", total, expected, fabs(total - expected));

    return(0);






}