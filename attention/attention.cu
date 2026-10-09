#include <cstdio>
#include <cstdlib>
#include <cmath>

inline void cudaCheck(cudaError_t err, const char* file, int line){
    if ( err != cudaSuccess ){
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

#define D 64

__global__ void attention(const float* Q, const float* K, const float * V, float* out, int N, int d){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;

   float scale = 1.0f/sqrtf((float)d);

   float m = -1e30f;
   float l = 0.0f;
   float o[D];
   for (int c = 0 ; c < d; c ++) o[c] = 0.0f;

   for (int j = 0; j< N; j ++){
        float score = 0.0f;
        for (int c = 0; c < d; c++){
            score += Q[i * d + c] * K[ j * d + c];
        }
        score *= scale;

        float m_new = fmaxf(m, score);
        float corr = expf(m - m_new);
        float p = expf(score - m_new);

        l = l * corr + p;
        for (int c = 0; c < d; c++){
            o[c] = o[c] * corr + p * V[j * d + c];
        }
        m = m_new;
        

    }
    for (int c = 0; c < d; c++){
        out[i * d +c] = o[c]/l;
    }


}


int main(){
    int N = 1024;
    int d = D;
    size_t bytes = (size_t)N * d * sizeof(float);

    float *h_Q = (float*)malloc(bytes);
    float *h_K = (float*)malloc(bytes);
    float *h_V = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N*d; i++) { 
        h_Q[i] = (float)((i * 7)  % 13) * 0.1f;
        h_K[i] = (float)((i * 11) % 17) * 0.1f;
        h_V[i] = (float)((i * 5)  % 19) * 0.1f;

     }  

    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    
    int threads =256;
    int blocks = (N+ threads -1) / threads;
    attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    attention<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.3f ms\n", ms);


    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    
    float scale = 1.0f / sqrtf((float)d);
    float* ref    = (float*)malloc((size_t)N * d * sizeof(float));
    float* scores = (float*)malloc((size_t)N * sizeof(float));  

    for (int i = 0; i < N; i++){                    
        for (int j = 0; j < N; j++){
            float s = 0.0f;
            for (int c = 0; c < d; c++) s += h_Q[i*d + c] * h_K[j*d + c];
            s *= scale;
            scores[j] = s;
            if (s > maxv) maxv = s;
        }
    
        float denom = 0.0f;
        for (int j = 0; j < N; j++) denom += expf(scores[j] - maxv);
        
        for (int c = 0; c < d; c++){
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += (expf(scores[j] - maxv) / denom) * h_V[j*d + c];
        ref[i*d + c] = acc;
        }
    }

    double max_err = 0.0;
    for (int idx = 0; idx < N*d; idx++)
        max_err = fmax(max_err, fabs(h_out[idx] - ref[idx]));
    printf("max error: %e  (expect < 1e-3)\n", max_err);
    free(ref); free(scores);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    free(h_Q); free(h_K); free(h_V); free(h_out);
    return 0;

}
