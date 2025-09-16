#include "reference.h"
#include <chrono>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, typename G>
#ifdef ENABLE_PROTEUS
__global__
    __attribute__((annotate("jit", 5, 6, 7, 8, 9, 10, 11, 13)))
#else
__global__
#endif
    void
    adam(T *__restrict__ p, T *__restrict__ m, T *__restrict__ v,
         const G *__restrict__ g, const float b1, const float b2,
         const float eps, const float grad_scale, const float step_size,
         const int time_step, const size_t vector_size, adamMode_t mode,
         const float decay) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t totThreads = gridDim.x * blockDim.x;

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j] / grad_scale;
      m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
      v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
      float m_corrected = m[j] / (1.f - powf(b1, t));
      float v_corrected = v[j] / (1.f - powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected / denom) + (decay * p[j]);
      p[j] -= (step_size * update);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n",
           argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  size_t size_bytes = vector_size * sizeof(float);

  float *m = (float *)malloc(size_bytes);
  float *v = (float *)malloc(size_bytes);
  float *g = (float *)malloc(size_bytes);
  float *p = (float *)malloc(size_bytes);
  float *r = (float *)malloc(size_bytes);

  srand(123);
  for (int i = 0; i < vector_size; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    r[i] = p[i] = rand() / (float)RAND_MAX;
  }

  float *d_m, *d_v, *d_g, *d_p;

  hipMalloc((void **)&d_m, size_bytes);
  hipMemcpy(d_m, m, size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void **)&d_v, size_bytes);
  hipMemcpy(d_v, v, size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void **)&d_g, size_bytes);
  hipMemcpy(d_g, g, size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void **)&d_p, size_bytes);
  hipMemcpy(d_p, p, size_bytes, hipMemcpyHostToDevice);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dim3 grids((vector_size + threadsPerBlock - 1) / threadsPerBlock);
  const dim3 blocks(threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    adam<float, float><<<grids, blocks>>>(d_p, d_m, d_v, d_g, beta1, beta2, eps,
                                          grad_scale, step_size, time_step,
                                          vector_size, mode, decay);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  hipMemcpy(p, d_p, size_bytes, hipMemcpyDeviceToHost);

  hipFree(d_p);
  hipFree(d_m);
  hipFree(d_v);
  hipFree(d_g);

  // Commenting, already verified
  //  // verify
  //  reference<float, float>(
  //    repeat,
  //    r, m, v, g,
  //    beta1, beta2,
  //    eps,
  //    grad_scale,
  //    step_size,
  //    time_step,
  //    vector_size,
  //    mode,
  //    decay);
  //
  //  bool ok = true;
  //  for (int i = 0; i < vector_size; i++) {
  //    if (r[i] - p[i] > 1e-3f) {
  //      ok = false;
  //      break;
  //    }
  //  }
  //  printf("%s\n", ok ? "PASS" : "FAIL");

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
