#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <memory>
#include <hip/hip_runtime.h>
#include <proteus/JitFrontend.hpp>
#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;
using namespace builtins::gpu;

#define TARGET "hip"

float* attention_host(float* key, float* value, float* query,
  const int n, const int d)
{
// intermediate
float* dot_product = (float*) malloc (n * sizeof(float));
float* score = (float*) malloc (n * sizeof(float));
// result
float* output = (float*) malloc (d * sizeof(float));

for (int i = 0; i < n; i++) {
float sum = 0;
for (int j = 0; j < d; j++)
sum += key[i * d + j] * query[j];
dot_product[i] = sum;
}

float sum = 0;
for (int i = 0; i < n; i++)
sum += expf(dot_product[i]);

for (int i = 0; i < n; i++)
score[i] = expf(dot_product[i]) / sum;

for (int j = 0; j < d; j++) {
float sum = 0;
for (int i = 0; i < n; i++)
sum += score[i] * value[i * d + j];
output[j] = sum;
}

free(dot_product);
free(score);
return output;
}

// Kernel 1: Compute dot products and accumulate exp_sum
static auto getAttentionKernel1(int n, int d) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*, float*)>("attention_kernel1");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &key = std::get<0>(Args);
    auto &query = std::get<1>(Args);
    auto &dot_product = std::get<2>(Args);
    auto &exp_sum = std::get<3>(Args);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Bdimx = F.callBuiltin(getBlockDimX);

      auto i = F.defVar<int>(Bidx * Bdimx + Tidx);
      auto Nvar = F.defRuntimeConst<int>(n);
      auto Dvar = F.defRuntimeConst<int>(d);
      auto Zero = F.defRuntimeConst<int>(0);
      auto One = F.defRuntimeConst<int>(1);

      F.beginIf(i < Nvar);
      {
        auto sum = F.defVar<float>(0.0f);
        auto j = F.declVar<int>("j");

        F.beginFor(j, Zero, Dvar, One);
        {
          auto keyIdx = i * Dvar + j;
          sum = sum + (key[keyIdx] * query[j]);
        }
        F.endFor();

        dot_product[i] = sum;

        auto expVal = expf(sum);
        F.atomicAdd(exp_sum, F.convert<float>(expVal));
      }
      F.endIf();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// Kernel 2: Compute scores (normalized with exp_sum)
static auto getAttentionKernel2(int n) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*)>("attention_kernel2");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &exp_sum = std::get<0>(Args);
    auto &dot_product = std::get<1>(Args);
    auto &score = std::get<2>(Args);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Bdimx = F.callBuiltin(getBlockDimX);

      auto i = F.defVar<int>(Bidx * Bdimx + Tidx);
      auto Nvar = F.defRuntimeConst<int>(n);

      F.beginIf(i < Nvar);
      {
        auto expVal = expf(dot_product[i]);
        auto expSumVal = F.defVar<float>(exp_sum[0]);
        score[i] = expVal / expSumVal;
      }
      F.endIf();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// Kernel 3: Compute output using scores and values
static auto getAttentionKernel3(int n, int d) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*)>("attention_kernel3");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &score = std::get<0>(Args);
    auto &value = std::get<1>(Args);
    auto &output = std::get<2>(Args);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Bdimx = F.callBuiltin(getBlockDimX);

      auto j = F.defVar<int>(Bidx * Bdimx + Tidx);
      auto Dvar = F.defRuntimeConst<int>(d);
      auto Nvar = F.defRuntimeConst<int>(n);
      auto Zero = F.defRuntimeConst<int>(0);
      auto One = F.defRuntimeConst<int>(1);

      F.beginIf(j < Dvar);
      {
        auto sum = F.defVar<float>(0.0f);
        auto i = F.declVar<int>("i");

        F.beginFor(i, Zero, Nvar, One);
        {
          auto valueIdx = i * Dvar + j;
          sum = sum + (score[i] * value[valueIdx]);
        }
        F.endFor();

        output[j] = sum;
      }
      F.endIf();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

float* attention_device(float* key, float* value, float* query,
                        const int n, const int d, const int repeat, const int verify)
{
  // input
  float *d_key;
  hipMalloc((void**)&d_key, n * d * sizeof(float));
  hipMemcpy(d_key, key, n * d * sizeof(float), hipMemcpyHostToDevice);

  float *d_value;
  hipMalloc((void**)&d_value, n * d * sizeof(float));
  hipMemcpy(d_value, value, n * d * sizeof(float), hipMemcpyHostToDevice);

  float *d_query;
  hipMalloc((void**)&d_query, d * sizeof(float));
  hipMemcpy(d_query, query, d * sizeof(float), hipMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  hipMalloc((void**)&d_dot_product, n * sizeof(float));

  float *d_exp_sum;
  hipMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  hipMalloc((void**)&d_output, d * sizeof(float));

  float *d_score;
  hipMalloc((void**)&d_score, n * sizeof(float));

  hipDeviceSynchronize();

  // Build and compile kernels
  auto [JitMod1, KernelHandle1] = getAttentionKernel1(n, d);
  JitMod1->compile();

  auto [JitMod2, KernelHandle2] = getAttentionKernel2(n);
  JitMod2->compile();

  auto [JitMod3, KernelHandle3] = getAttentionKernel3(n, d);
  JitMod3->compile();

  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    if(verify) {
      hipMemset(d_exp_sum, 0, 4);
    }

    KernelHandle1.launch(
      {static_cast<unsigned int>((n+255)/256), 1u, 1u},
      {256u, 1u, 1u},
      0, nullptr,
      d_key, d_query, d_dot_product, d_exp_sum);

    KernelHandle2.launch(
      {static_cast<unsigned int>((n+255)/256), 1u, 1u},
      {256u, 1u, 1u},
      0, nullptr,
      d_exp_sum, d_dot_product, d_score);

    KernelHandle3.launch(
      {static_cast<unsigned int>((d+255)/256), 1u, 1u},
      {256u, 1u, 1u},
      0, nullptr,
      d_score, d_value, d_output);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);

  hipMemcpy(output, d_output, d * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_score);
  hipFree(d_value);
  hipFree(d_output);
  hipFree(d_key);
  hipFree(d_dot_product);
  hipFree(d_exp_sum);
  hipFree(d_query);
  return output;
}

int main(int argc, char* argv[]) {
  proteus::init();

  if (argc != 4 && argc != 5) {
    printf("Usage: %s <rows> <columns> <repeat> [verify]\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);
  const int verify = (argc == 5) ? atoi(argv[4]) : 0;

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  if (verify) {
  for (int i = 0; i < n * d; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
      query[i % d] = dist(gen);
    }
  }

  float* dout = attention_device(key, value, query, n, d, r, verify);

  if (verify) {
    float* hout = attention_host(key, value, query, n, d);

    float rmse = 0;
    for (int i = 0; i < d; i++) {
      rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
    }
    printf("RMSE = %f\n", sqrtf(rmse / d));

    free(hout);
  }

  free(key);
  free(value);
  free(query);
  free(dout);

  proteus::finalize();
  return 0;
}
