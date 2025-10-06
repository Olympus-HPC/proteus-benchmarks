#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>

#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace proteus;
using namespace builtins::gpu;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif


typedef enum {
  ADAM_MODE_0 = 0, // eps under square root
  ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;



auto createJitModuleSpecial(float _b1, float _b2, float _eps, float _grad_scale,
                            float _step_size, int _time_step,
                            size_t _vector_size, int _mode, float _decay) {
  auto J = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      J->addKernel<void(float *, float *, float *, float *)>("adam");
  auto &F = KernelHandle.F;
  auto [p, m, v, g] = F.getArgs();


  F.beginFunction();
  {
    auto [b1, b2, eps, grad_scale, step_size, time_step, vector_size, mode,
          decay] = F.defRuntimeConsts(_b1, _b2, _eps, _grad_scale, _step_size,
                                      _time_step, _vector_size, _mode, _decay);

    
    auto &i = F.declVar<size_t>("i");
    auto &totThreads = F.declVar<size_t>("totThreads");
    auto &j = F.declVar<size_t>("j");
    auto &t = F.declVar<int>("t");
    auto &inc1 = F.defRuntimeConst<int>(1);

    i = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    totThreads = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    F.beginFor(j, i, vector_size, totThreads);
    {
      auto &lim = F.declVar<int>("lim");
      t = 1;
      lim = time_step;
      F.beginFor(t, t, lim, inc1);
      {
        auto &scaled_grad = F.declVar<float>("scale_grad");
        scaled_grad = g[j] / grad_scale;

        m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;

        auto &m_corrected = F.declVar<float>("m_corrected");
        auto &v_corrected = F.declVar<float>("v_corrected");
        m_corrected = m[j] / (1.f - powf(b1, t));
        v_corrected = v[j] / (1.f - powf(b2, t));

        auto &denom = F.declVar<float>("denom");
        F.beginIf(mode == 0);
        { denom = sqrtf(v_corrected + eps); }
        F.endIf();

        F.beginIf(mode == 1);
        { denom = sqrtf(v_corrected) + eps; }
        F.endIf();

        auto &update = F.declVar<float>("update");
        update = (m_corrected / denom) + (decay * p[j]);

        p[j] -= (step_size * update);
      }
      F.endFor();
    }
    F.endFor();
    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}

int main(int argc, char *argv[]) {
  if (argc < 4 || argc > 5) {
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

  auto [J, KernelHandle] =
      createJitModuleSpecial(beta1, beta2, eps, grad_scale, step_size,
                             time_step, vector_size, mode, decay);

  J->compile();

  hipDeviceSynchronize();


  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // adam<float, float><<<grids, blocks>>>(d_p, d_m, d_v, d_g, beta1, beta2,
    // eps,
    //                                       grad_scale, step_size, time_step,
    //                                       vector_size, mode, decay);

    KernelHandle.launch({grids.x, 1, 1}, {blocks.x, 1, 1}, 0, 0,
                                    d_p, d_m, d_v, d_g);
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

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
