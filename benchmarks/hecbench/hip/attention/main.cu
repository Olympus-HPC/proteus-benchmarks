#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "kernels.h"

float* attention_host(const float* key, const float* value, const float* query,
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




float* attention_device(const float* key, const float* value, const float* query,
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

  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    if(verify) {
      hipMemset(d_exp_sum, 0, 4);
    }
    attention_kernel1<<<(n+255)/256, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
    attention_kernel2<<<(n+255)/256, 256>>>(d_exp_sum, d_dot_product, d_score, n);
    attention_kernel3<<<(d+255)/256, 256>>>(d_score, d_value, d_output, n, d);
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
  return 0;
}
