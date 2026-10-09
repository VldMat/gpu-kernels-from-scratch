#include <cstdio>
#include <cstdlib>
#include <cmath>



inline void cudaCheck( cudaError_t err, const char* file, int line){
    if ( err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

#define COLS 1024

__global__ void online_softmax(const float* in, float* out, int cols){
    int rows = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float sm[COLS];
    __shared__ float sl[COLS];

    float x = in[rows*cols + tid];
    float maxv;
    sm[tid] = x;
    sl[tid] = 1.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (tid < s){

            maxv = fmaxf(sm[tid], sm[tid+s]);
            sl[tid] = sl[tid] * expf(sm[tid] - maxv) + sl[tid+s] * (expf(sm[tid+s]-maxv));
            sm[tid] = maxv;
        }
        __syncthreads();
    }
    out[rows *cols + tid] = expf(x-sm[0]) / sl[0];


}

int main(){
    int rows =1024, cols =1024;
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = (float)((i *13) %17) * 1.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads = cols;
    int blocks =rows;
    online_softmax<<<blocks, threads>>>(d_in, d_out, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    online_softmax<<<blocks, threads>>>(d_in, d_out,  cols);   // warm-up run (first launch has extra overhead)
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    online_softmax<<<blocks, threads>>>(d_in, d_out, cols);   // the timed run
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.3f ms\n", ms);


    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));


    double max_err = 0.0;
    for (int r = 0; r < rows; r++){
        double rowsum = 0.0;
        for (int c = 0; c < cols; c++)
            rowsum += h_out[r * cols + c];        // row-major: r*cols + c
        max_err = fmax(max_err, fabs(rowsum - 1.0));  // how far this row is from 1.0
    }
    printf("max row-sum error: %e  (expect < 1e-4)\n", max_err);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}



