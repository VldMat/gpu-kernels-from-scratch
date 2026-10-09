#include <cstdio>
#include <cstdlib>
#include <cmath>

inline void cudaCheck(cudaError_t err, const char* file, int line){
    if( err != cudaSuccess){
        fprintf( stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

#define D 64
#define BC 64
#define BQ 64

__global__ void attention_tiled(const float* Q, const float* K, 
                                const float* V, float* out, int N, int d){
        int t =threadIdx.x;
        int row = blockIdx.x * BQ + t;
        if(row >= N) return;

        float scale = 1.0f / sqrtf((float)d);

        __shared__ float Ks[BC][D];
        __shared__ float Vs[BC][D];

        float qi[D];
        for (int  c = 0; c < d; c++) qi[c] = Q[row * d + c];

        float m = -1e30f;
        float l = 0.0f;
        float o[D];
        for (int c = 0; c < d; c++) o[c] = 0.0f;

        for (int tile = 0; tile < N / blockDim.x; tile ++){
            int krow = tile * BC + t;
            for (int c = 0; c < d; c++){
                Ks[t][c] = K[krow * d + c];
                Vs[t][c] = V[krow * d + c];

            }
            __syncthreads();
            for (int jj = 0; jj < BC; jj ++){
                float score = 0.0f;
                for ( int c = 0; c<d; c++) score += qi[c] * Ks[jj][c];
                score *= scale;

                float m_next = fmaxf(m, score);
                float corr = expf(m - m_next);
                float p = expf(score - m_next);
                l = l * corr + p;
                for (int c=0; c < d; c++) o[c] = o[c] * corr + p * Vs[jj][c];
                m = m_next;

            }
            __syncthreads();
        }
        for (int c = 0; c< d; c++) out[row * d + c] = o[c] / l;

}



int main(){
    int N = 1 << 10;
    int d = D;
    size_t bytes = (size_t)N * d * sizeof(float);

    float *h_Q = (float*)malloc(bytes);
    float *h_K = (float*)malloc(bytes);
    float *h_V = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N * d; i++){
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

    int threads = BQ;
    int blocks = N / BQ;
    attention_tiled<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    attention_tiled<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    attention_tiled<<<blocks, threads>>>(d_Q, d_K, d_V, d_out, N, d);   
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
    float* scores = (float*)malloc((size_t)N * sizeof(float));   // reused per row

    for (int i = 0; i < N; i++){                    // each query row
    // 1) all scores for row i, and the row max (for stable softmax)
        float maxv = -1e30f;
        for (int j = 0; j < N; j++){
            float s = 0.0f;
            for (int c = 0; c < d; c++) s += h_Q[i*d + c] * h_K[j*d + c];
            s *= scale;
            scores[j] = s;
            if (s > maxv) maxv = s;
        }
    // 2) softmax denominator
        float denom = 0.0f;
        for (int j = 0; j < N; j++) denom += expf(scores[j] - maxv);
        // 3) output row = Σ_j softmax_weight(j) * V[j]
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