/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <utility>
#include <hip/hip_runtime.h>
#define TARGET "hip"

#include <proteus/JitFrontend.hpp>
#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;
using namespace builtins::gpu;

#define TILE_WIDTH 16

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

void verify (const float* Y, float* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

// JIT kernel builder for conv3d_s1: grid(N, M, Z)
static auto getConv3dS1Kernel(int C_, int M_, int K_, int Hin_, int Win_, int Hout_, int Wout_, int W_grid_) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*)>("conv3d_s1");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &X = std::get<0>(Args);
    auto &W = std::get<1>(Args);
    auto &Y = std::get<2>(Args);

    F.beginFunction();
    {
      auto &C = F.defRuntimeConst(C_);
      auto &M = F.defRuntimeConst(M_);
      auto &K = F.defRuntimeConst(K_);
      auto &Hin = F.defRuntimeConst(Hin_);
      auto &Win = F.defRuntimeConst(Win_);
      auto &Hout = F.defRuntimeConst(Hout_);
      auto &Wout = F.defRuntimeConst(Wout_);
      auto &WGrid = F.defRuntimeConst(W_grid_);

      // Constants
      auto &TileWidth = F.defRuntimeConst(TILE_WIDTH);
      auto &Zero = F.defRuntimeConst(0);
      auto &One = F.defRuntimeConst(1);

      auto &n = F.callBuiltin(getBlockIdX);
      auto &m = F.callBuiltin(getBlockIdY);
      auto &h = F.callBuiltin(getBlockIdZ) / WGrid * TileWidth + F.callBuiltin(getThreadIdY);
      auto &w = F.callBuiltin(getBlockIdZ) % WGrid * TileWidth + F.callBuiltin(getThreadIdX);

      F.beginIf(h >= Hout);
      { F.ret();}
      F.endIf();
      F.beginIf(w >= Wout);
      { F.ret();}
      F.endIf();

      auto &s = F.defVar(0.0f);
      auto &c = F.declVar<int>("c");
      auto &p = F.declVar<int>("p");
      auto &q = F.declVar<int>("q");
      F.buildLoopNest(
        F.forLoop({c, Zero, C, One}),
        F.forLoop({p, Zero, K, One}),
        F.forLoop({q, Zero, K, One}, [&]() {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        })
      ).emit();

      Y[OI(n, m, h, w)] = s;

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// JIT kernel builder for conv3d_s2: grid(M, Z, N)
static auto getConv3dS2Kernel(int C_, int M_, int K_, int Hin_, int Win_, int Hout_, int Wout_, int W_grid_) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*)>("conv3d_s2");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &X = std::get<0>(Args);
    auto &W = std::get<1>(Args);
    auto &Y = std::get<2>(Args);

    F.beginFunction();
    {
      auto &C = F.defRuntimeConst(C_);
      auto &M = F.defRuntimeConst(M_);
      auto &K = F.defRuntimeConst(K_);
      auto &Hin = F.defRuntimeConst(Hin_);
      auto &Win = F.defRuntimeConst(Win_);
      auto &Hout = F.defRuntimeConst(Hout_);
      auto &Wout = F.defRuntimeConst(Wout_);
      auto &WGrid = F.defRuntimeConst(W_grid_);

      // Constants
      auto &TileWidth = F.defRuntimeConst(TILE_WIDTH);
      auto &Zero = F.defRuntimeConst(0);
      auto &One = F.defRuntimeConst(1);

      auto &m = F.callBuiltin(getBlockIdX);
      auto &h = F.callBuiltin(getBlockIdY) / WGrid * TileWidth + F.callBuiltin(getThreadIdY);
      auto &w = F.callBuiltin(getBlockIdY) % WGrid * TileWidth + F.callBuiltin(getThreadIdX);
      auto &n = F.callBuiltin(getBlockIdZ);

      F.beginIf(h >= Hout);
      { F.ret();}
      F.endIf();
      F.beginIf(w >= Wout);
      { F.ret();}
      F.endIf();

      auto &s = F.defVar(0.0f);
      auto &c = F.declVar<int>("c");
      auto &p = F.declVar<int>("p");
      auto &q = F.declVar<int>("q");
      F.buildLoopNest(
        F.forLoop({c, Zero, C, One}),
        F.forLoop({p, Zero, K, One}),
        F.forLoop({q, Zero, K, One}, [&]() {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        })
      ).emit();

      Y[OI(n, m, h, w)] = s;

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// JIT kernel builder for conv3d_s3: grid(Z, N, M)
static auto getConv3dS3Kernel(int C_, int M_, int K_, int Hin_, int Win_, int Hout_, int Wout_, int W_grid_) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*, float*, float*)>("conv3d_s3");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &X = std::get<0>(Args);
    auto &W = std::get<1>(Args);
    auto &Y = std::get<2>(Args);

    F.beginFunction();
    {
      auto &C = F.defRuntimeConst(C_);
      auto &M = F.defRuntimeConst(M_);
      auto &K = F.defRuntimeConst(K_);
      auto &Hin = F.defRuntimeConst(Hin_);
      auto &Win = F.defRuntimeConst(Win_);
      auto &Hout = F.defRuntimeConst(Hout_);
      auto &Wout = F.defRuntimeConst(Wout_);
      auto &WGrid = F.defRuntimeConst(W_grid_);

      // Constants
      auto &TileWidth = F.defRuntimeConst(TILE_WIDTH);
      auto &Zero = F.defRuntimeConst(0);
      auto &One = F.defRuntimeConst(1);

      auto &h = F.callBuiltin(getBlockIdX) / WGrid * TileWidth + F.callBuiltin(getThreadIdY);
      auto &w = F.callBuiltin(getBlockIdX) % WGrid * TileWidth + F.callBuiltin(getThreadIdX);
      auto &n = F.callBuiltin(getBlockIdY);
      auto &m = F.callBuiltin(getBlockIdZ);

      F.beginIf(h >= Hout);
      { F.ret();}
      F.endIf();
      F.beginIf(w >= Wout);
      { F.ret();}
      F.endIf();

      auto &s = F.defVar(0.0f);
      auto &c = F.declVar<int>("c");
      auto &p = F.declVar<int>("p");
      auto &q = F.declVar<int>("q");
      F.buildLoopNest(
        F.forLoop({c, Zero, C, One}),
        F.forLoop({p, Zero, K, One}),
        F.forLoop({q, Zero, K, One}, [&]() {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        })
      ).emit();

      Y[OI(n, m, h, w)] = s;

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// Hin = Hout-1+K; max(h+p) is Hin - 1 as max(h) = Hout-1 and max(p) = K-1
void reference(const float * __restrict__ X,
               const float * __restrict__ W,
                     float * __restrict__ Y,
               const int N,
               const int M,
               const int C,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout)
{
  for(int n = 0; n < N; n++)
    for(int m = 0; m < M; m++)
      for(int h = 0; h < Hout; h++)
        for(int w = 0; w < Wout; w++) {
          Y[OI(n, m, h, w)] = 0;
          for(int c = 0; c < C; c++)
            for(int p = 0; p < K; p++)
              for(int q = 0; q < K; q++)
                Y[OI(n, m, h, w)] += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
}

void conv3D(const int N, const int C, const int M, const int Win, const int Hin, const int K, const int repeat, const int do_verify)
{
  const int Hout = Hin-K+1;
  const int Wout = Win-K+1;

  size_t X_size = N * C * Hin * Win;
  size_t W_size = M * C * K * K;
  size_t Y_size = N * M * Hout * Wout;
  size_t X_bytes = X_size * sizeof(float);
  size_t W_bytes = W_size * sizeof(float);
  size_t Y_bytes = Y_size * sizeof(float);

  float *X, *W, *Y, *Y_ref;
  X = (float *)malloc(X_bytes); // input
  W = (float *)malloc(W_bytes); // filter
  Y = (float *)malloc(Y_bytes); // output

  srand(123);


  if (do_verify) {
    for (size_t i = 0; i < W_size; i++) W[i] = rand() % 31;
    for (size_t i = 0; i < X_size; i++) X[i] = rand() % 13;

    for (size_t i = 0; i < Y_size; i++) {
      Y[i] = -1;
    }
    Y_ref = (float *)malloc(Y_bytes);
    for (size_t i = 0; i < Y_size; i++) {
      Y_ref[i] = -1;
    }
    reference(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);
  }

  float *dX, *dW, *dY;
  hipMalloc((void **)&dX, X_bytes);
  hipMalloc((void **)&dW, W_bytes);
  hipMalloc((void **)&dY, Y_bytes);

  hipMemcpy(dX, X, X_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dW, W, W_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dY, Y, Y_bytes, hipMemcpyHostToDevice);

  int W_grid = (Wout + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (Hout + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);
  printf("3D grid dimensions: N=%d M=%d Z=%d\n", N, M, Z);

  // Build and compile kernels
  auto [JitMod1, KernelHandle1] = getConv3dS1Kernel(C, M, K, Hin, Win, Hout, Wout, W_grid);
  JitMod1->compile();

  auto [JitMod2, KernelHandle2] = getConv3dS2Kernel(C, M, K, Hin, Win, Hout, Wout, W_grid);
  JitMod2->compile();

  auto [JitMod3, KernelHandle3] = getConv3dS3Kernel(C, M, K, Hin, Win, Hout, Wout, W_grid);
  JitMod3->compile();

  hipDeviceSynchronize();

  // Test conv3d_s1 with grid(N, M, Z)
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelHandle1.launch(
      {static_cast<unsigned int>(N), static_cast<unsigned int>(M), static_cast<unsigned int>(Z)},
      {TILE_WIDTH, TILE_WIDTH, 1u},
      0, nullptr,
      dX, dW, dY);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
    verify(Y, Y_ref, Y_size);
  }

  // Test conv3d_s2 with grid(M, Z, N)
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelHandle2.launch(
      {static_cast<unsigned int>(M), static_cast<unsigned int>(Z), static_cast<unsigned int>(N)},
      {TILE_WIDTH, TILE_WIDTH, 1u},
      0, nullptr,
      dX, dW, dY);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
    verify(Y, Y_ref, Y_size);
  }

  // Test conv3d_s3 with grid(Z, N, M)
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    KernelHandle3.launch(
      {static_cast<unsigned int>(Z), static_cast<unsigned int>(N), static_cast<unsigned int>(M)},
      {TILE_WIDTH, TILE_WIDTH, 1u},
      0, nullptr,
      dX, dW, dY);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  if (do_verify) {
    hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
    verify(Y, Y_ref, Y_size);
  }

  free(X);
  free(W);
  free(Y);
  if (do_verify) {
    free(Y_ref);
  }
  hipFree(dX);
  hipFree(dW);
  hipFree(dY);
}

int main(int argc, char* argv[]) {
  proteus::init();

  if (argc != 8 && argc != 9) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat> [verify (0 or 1, default 0)]\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int W = atoi(argv[4]);
  int H = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);
  int verify = (argc == 9) ? atoi(argv[8]) : 0;

  printf("3D convolution (FP32)\n");
  printf("\n========== Warmup start ==========\n");
  conv3D(N, C, M, W, H, K, 1000, verify);
  printf("\n========== Warmup done ==========\n");
  conv3D(N, C, M, W, H, K, repeat, verify);

  proteus::finalize();
  return 0;
}
