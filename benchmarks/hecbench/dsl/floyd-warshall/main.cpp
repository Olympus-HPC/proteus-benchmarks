#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#if !defined(PROTEUS_ENABLE_HIP) && !defined(PROTEUS_ENABLE_CUDA)
#define PROTEUS_ENABLE_HIP 1
#endif

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

constexpr unsigned int MAXDISTANCE = 200;

using namespace proteus;
using namespace builtins::gpu;

// Map RNG output to [0, MAXDISTANCE] and zero the diagonal (2D launch)
extern "C" __global__ void initRandomMatrix2D(unsigned int* __restrict__ buf, const unsigned int numNodes) {
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= numNodes || y >= numNodes) return;
  unsigned int idx = y * numNodes + x;
  unsigned int v = buf[idx] % (MAXDISTANCE + 1);
  if (x == y) v = 0u;
  buf[idx] = v;
}

// Reference CPU implementation for verification
static void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
                                      unsigned int * pathMatrix,
                                      unsigned int numNodes)
{
  unsigned int width = numNodes;
  for (unsigned int k = 0; k < numNodes; ++k) {
    for (unsigned int y = 0; y < numNodes; ++y) {
      unsigned int yXwidth = y * numNodes;
      for (unsigned int x = 0; x < numNodes; ++x) {
        unsigned int distanceYtoX = pathDistanceMatrix[yXwidth + x];
        unsigned int distanceYtoK = pathDistanceMatrix[yXwidth + k];
        unsigned int distanceKtoX = pathDistanceMatrix[k * width + x];
        unsigned int indirectDistance = distanceYtoK + distanceKtoX;
        if (indirectDistance < distanceYtoX) {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x] = k;
        }
      }
    }
  }
}

static auto createJitModuleSpecial(unsigned int _numNodes) {
  auto J = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = J->addKernel<void(unsigned int*, unsigned int*, unsigned int, unsigned int)>("floydWarshallPass");
  auto &F = KernelHandle.F;
  auto [pathDistanceBuffer, pathBuffer, numNodes, pass] = F.getArgs();

  F.beginFunction();
  {
    // Bake only numNodes as a runtime constant; pass is a dynamic kernel arg
    auto rcNumNodes = F.defRuntimeConst(_numNodes);

    auto xValue = F.declVar<unsigned int>("xValue");
    auto yValue = F.declVar<unsigned int>("yValue");
    auto k = F.declVar<unsigned int>("k");
    auto oldWeight = F.declVar<unsigned int>("oldWeight");
    auto tempWeight = F.declVar<unsigned int>("tempWeight");

    // 2D thread/block indices
    auto tidx = F.callBuiltin(getThreadIdX);
    auto tidy = F.callBuiltin(getThreadIdY);
    auto bidx = F.callBuiltin(getBlockIdX);
    auto bidy = F.callBuiltin(getBlockIdY);
    auto bdimx = F.callBuiltin(getBlockDimX);
    auto bdimy = F.callBuiltin(getBlockDimY);

    xValue = bidx * bdimx + tidx;
    yValue = bidy * bdimy + tidy;
    k = pass;

    oldWeight = pathDistanceBuffer[yValue * rcNumNodes + xValue];
    tempWeight = pathDistanceBuffer[yValue * rcNumNodes + k] +
                 pathDistanceBuffer[k * rcNumNodes + xValue];

    F.beginIf(tempWeight < oldWeight);
    {
      pathDistanceBuffer[yValue * rcNumNodes + xValue] = tempWeight;
      pathBuffer[yValue * rcNumNodes + xValue] = k;
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}

int main(int argc, char **argv) {
  proteus::init();
  if (argc != 4 && argc != 5) {
    std::printf("Usage: %s <number of nodes> <iterations> <block size> [verify (0 or 1, default 0)]\n", argv[0]);
    return 1;
  }

  unsigned int numNodes = static_cast<unsigned int>(std::atoi(argv[1]));
  unsigned int numIterations = static_cast<unsigned int>(std::atoi(argv[2]));
  unsigned int blockSize = static_cast<unsigned int>(std::atoi(argv[3]));
  int do_verify = (argc == 5) ? std::atoi(argv[4]) : 0;

  const size_t matrixSizeBytes = static_cast<size_t>(numNodes) * static_cast<size_t>(numNodes) * sizeof(unsigned int);

  unsigned int *pathDistanceMatrix = nullptr;
  unsigned int *pathMatrix = nullptr;
  unsigned int *verificationPathDistanceMatrix = nullptr;
  unsigned int *verificationPathMatrix = nullptr;

  if (do_verify) {
    pathDistanceMatrix = (unsigned int *)std::malloc(matrixSizeBytes);
    pathMatrix = (unsigned int *)std::malloc(matrixSizeBytes);
    assert(pathDistanceMatrix && pathMatrix);

    // Initialize path matrix
    for (unsigned int i = 0; i < numNodes; ++i) {
      for (unsigned int j = 0; j < i; ++j) {
        pathMatrix[i * numNodes + j] = i;
        pathMatrix[j * numNodes + i] = j;
      }
      pathMatrix[i * numNodes + i] = i;
    }

    verificationPathDistanceMatrix = (unsigned int *)std::malloc(matrixSizeBytes);
    verificationPathMatrix = (unsigned int *)std::malloc(matrixSizeBytes);
    assert(verificationPathDistanceMatrix && verificationPathMatrix);
    std::memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);
  }

  if (blockSize * blockSize > 256u) {
    blockSize = 16u;
  }

  // 2D launch configuration
  const unsigned int gridX = (numNodes + blockSize - 1u) / blockSize;
  const unsigned int gridY = (numNodes + blockSize - 1u) / blockSize;

  unsigned int *pathDistanceBuffer = nullptr;
  unsigned int *pathBuffer = nullptr;
  (void)hipMalloc(reinterpret_cast<void **>(&pathDistanceBuffer), matrixSizeBytes);
  (void)hipMalloc(reinterpret_cast<void **>(&pathBuffer), matrixSizeBytes);

  // JIT compile kernel specialized for this numNodes
  auto [J, KernelHandle] = createJitModuleSpecial(static_cast<unsigned int>(numNodes));
  J->compile();

  // hipRAND generator
  hiprandGenerator_t gen;
  (void)hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  (void)hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  float total_time_ns = 0.0f;

  for (unsigned int n = 0; n < numIterations; n++) {
    // Generate matrix on device using hipRAND
    (void)hiprandGenerate(gen, pathDistanceBuffer, numNodes * numNodes);
    // Map to [0, MAXDISTANCE] and zero diagonal
    initRandomMatrix2D<<<dim3(gridX, gridY, 1), dim3(blockSize, blockSize, 1)>>>(pathDistanceBuffer, numNodes);

    if (do_verify && n == numIterations - 1) {
      // Save initial matrix for CPU reference on last iteration
      (void)hipMemcpy(verificationPathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, hipMemcpyDeviceToHost);
    }

    (void)hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < numNodes; i++) {
      (void)KernelHandle.launch({gridX, gridY, 1U}, {blockSize, blockSize, 1U}, 0, nullptr,
                                pathDistanceBuffer, pathBuffer, static_cast<unsigned>(numNodes), static_cast<unsigned>(i));
    }

    (void)hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time_ns += static_cast<float>(time_ns);
  }

  (void)hiprandDestroyGenerator(gen);

  std::printf("Average kernel execution time %f (s)\n", (total_time_ns * 1e-9f) / static_cast<float>(numIterations));

  if (do_verify) {
    (void)hipMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, hipMemcpyDeviceToHost);
  }

  (void)hipFree(pathDistanceBuffer);
  (void)hipFree(pathBuffer);

  // verify
  if (do_verify) {
    floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
    if (std::memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0) {
      std::printf("PASS\n");
    } else {
      std::printf("FAIL\n");
      if (numNodes <= 8) {
        for (unsigned int i = 0; i < numNodes; i++) {
          for (unsigned int j = 0; j < numNodes; j++)
            std::printf("host: %u ", verificationPathDistanceMatrix[i*numNodes+j]);
          std::printf("\n");
        }
        for (unsigned int i = 0; i < numNodes; i++) {
          for (unsigned int j = 0; j < numNodes; j++)
            std::printf("device: %u ", pathDistanceMatrix[i*numNodes+j]);
          std::printf("\n");
        }
      }
    }
  }

  if (do_verify) {
    std::free(pathDistanceMatrix);
    std::free(pathMatrix);
    std::free(verificationPathDistanceMatrix);
    std::free(verificationPathMatrix);
  }

  proteus::finalize();
  return 0;
}
