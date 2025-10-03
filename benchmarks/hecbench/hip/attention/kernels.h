#ifdef ENABLE_PROTEUS
__attribute__((annotate("jit", 5, 6)))
#endif
__global__
void attention_kernel1 (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

#ifdef ENABLE_PROTEUS
__attribute__((annotate("jit", 4)))
#endif
__global__
void attention_kernel2 (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    score[i] = __expf(dot_product[i]) / exp_sum[0];
}

#ifdef ENABLE_PROTEUS
__attribute__((annotate("jit", 4, 5)))
#endif
__global__
void attention_kernel3 (
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}
