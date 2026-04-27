

#include <math.h>
#include <omp.h>
#include <string.h>



double smoothness_loss_omp(
    const float* scores,     /* [N] anomaly scores */
    const long* row,         /* [E] source indices */
    const long* col,         /* [E] destination indices */
    int E)                   /* number of edges */
{
    double total = 0.0;

    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int e = 0; e < E; e++) {
        float diff = scores[row[e]] - scores[col[e]];
        total += (double)(diff * diff);
    }

    return total / (double)E;  /* mean, consistent with CUDA version */
}



void gat_attention_omp(
    const float* Wh,          /* [N, D] transformed features */
    const long* row,          /* [E] source indices */
    const long* col,          /* [E] destination indices */
    const float* attn_vec,    /* [2*D] attention vector */
    float* e_out,             /* [E] output logits */
    int E, int D,
    float negative_slope)
{
    /* [Lec 2] Each thread handles a subset of edges.
     * No synchronization needed — each edge writes to a unique output. */
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        long src = row[e];
        long dst = col[e];
        float val = 0.0f;

        for (int d = 0; d < D; d++) {
            val += Wh[dst * D + d] * attn_vec[d];
            val += Wh[src * D + d] * attn_vec[D + d];
        }

        /* LeakyReLU */
        e_out[e] = (val > 0.0f) ? val : val * negative_slope;
    }
}



void neighbor_aggregation_omp(
    const float* Wh,          /* [N, D] source features */
    const float* alpha,       /* [E] attention weights */
    const long* row,          /* [E] source indices */
    const long* col,          /* [E] destination indices */
    float* out,               /* [N, D] output (must be zero-initialized) */
    int E, int D)
{
    /* [Lec 2] Parallelize over edges.
     * [Lec 20] The #pragma omp atomic on the output update handles the
     * many-to-one write pattern.  This is analogous to atomicAdd in CUDA
     * but with CPU-level atomic instructions.
     * Note: atomic is needed because multiple edges may have the same dst. */
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        long src = row[e];
        long dst = col[e];
        float a  = alpha[e];

        for (int d = 0; d < D; d++) {
            float contribution = a * Wh[src * D + d];
            #pragma omp atomic
            out[dst * D + d] += contribution;
        }
    }
}



void matmul_omp(
    const float* A,   /* [M, K] */
    const float* B,   /* [K, N] */
    float* C,         /* [M, N] — must be zero-initialized */
    int M, int K, int N)
{
    /* [Lec 2] Parallelize over rows of C.
     * Each thread computes a full row of the output independently. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}


/* Helper: returns the number of OpenMP threads available */
int get_num_threads(void) {
    int n;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}
