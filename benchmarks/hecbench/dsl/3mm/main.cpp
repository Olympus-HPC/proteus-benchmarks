#include <proteus/JitFrontend.hpp>
#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitInterface.hpp>

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <iostream>
#include <memory>
#include <utility>
#include <chrono>

using namespace proteus;
using namespace builtins::gpu;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#include <hip/hip_runtime.h>
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetErrorString cudaGetErrorString
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif


// Naive tiled JIT kernel (C = A x B) for square N x N, double.
static auto getMatmulKernel(int N, int TileSize) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(double *, double *, double *)>("tiled_matmul");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Tidy = F.callBuiltin(getThreadIdY);
      auto Bidy = F.callBuiltin(getBlockIdY);

      auto Row = Bidy * TileSize + Tidy;
      auto Col = Bidx * TileSize + Tidx;

      auto K = F.declVar<int>("K");
      auto Zero = F.defRuntimeConst(0);
      auto One = F.defRuntimeConst(1);
      auto Nvar = F.defRuntimeConst(N);
      auto Accum = F.defVar(0.0);

      F.beginFor(K, Zero, Nvar, One);
      {
        auto AVal = A[Row * N + K];
        auto BVal = B[K * N + Col];
        Accum = Accum + (AVal * BVal);
      }
      F.endFor();

      auto CIdx = Row * N + Col;
      C[CIdx] = Accum;

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

// Register + shared-memory tiled JIT kernel (C = A x B) for square N x N, double.
// Configuration mirrors the HIP kernel: 64x64 block tile, 16x16 threads,
// 4x4 per-thread micro-tile, K tile = 8 (defaults; overridable via CLI).
static auto getRegSharedTiledMatmulKernel(int N, int blockTileM, int blockTileN, int kTile, int regTileM, int regTileN) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(double *, double *, double *)>("reg_shared_tiled_matmul");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    auto AsTile = F.declVar<double[]>(blockTileM * kTile, AddressSpace::SHARED);
    auto BsTile = F.declVar<double[]>(kTile * blockTileN, AddressSpace::SHARED);

    auto Areg = F.declVar<double[]>(regTileM);
    auto Breg = F.declVar<double[]>(regTileN);
    auto Creg = F.declVar<double[]>(regTileM * regTileN);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Tidy = F.callBuiltin(getThreadIdY);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Bidy = F.callBuiltin(getBlockIdY);

      auto Nvar = F.defRuntimeConst(N);
      auto Two = F.defRuntimeConst(2);
      auto Zero = F.defRuntimeConst(0);
      auto One = F.defRuntimeConst(1);
      auto RegTileM = F.defRuntimeConst(regTileM);
      auto RegTileN = F.defRuntimeConst(regTileN);
      auto BlockTileM = F.defRuntimeConst(blockTileM);
      auto BlockTileN = F.defRuntimeConst(blockTileN);
      auto KTile = F.defRuntimeConst(kTile);
      auto ThreadsX = F.defRuntimeConst(blockTileN / regTileN);

      auto BlockRow = Bidy * BlockTileM;
      auto BlockCol = Bidx * BlockTileN;

      auto Row0 = BlockRow + Tidy * RegTileM;
      auto Col0 = BlockCol + Tidx * RegTileN;

      {
        auto I = F.declVar<int>("i");
        auto J = F.declVar<int>("j");
        F.forLoop<int>({I, Zero, RegTileM, One}, [&]() {
          F.forLoop<int>({J, Zero, RegTileN, One}, [&]() {
            auto Cidx = I * RegTileN + J;
            Creg[Cidx] = 0.0;
          }).emit();
        }).emit();
      }

      auto KBase = F.declVar<int>("KBase");
      F.forLoop<int>({KBase, Zero, Nvar, KTile}, [&]() {
        auto Tid = Tidy * ThreadsX + Tidx;

        auto APlaneSize = BlockTileM * KTile;
        auto AIdx1 = Tid * Two + One;
        auto AIdx0 = Tid * Two + Zero;
        auto ARow0 = AIdx0 / KTile;
        auto ACol0 = AIdx0 % KTile;
        auto ARow1 = AIdx1 / KTile;
        auto ACol1 = AIdx1 % KTile;
        auto AsIdx0 = ARow0 * KTile + ACol0;
        auto AsIdx1 = ARow1 * KTile + ACol1;
        auto AGlobIdx0 = (BlockRow + ARow0) * N + (KBase + ACol0);
        auto AGlobIdx1 = (BlockRow + ARow1) * N + (KBase + ACol1);
        F.beginIf(AIdx0 < APlaneSize);
        {
          AsTile[AsIdx0] = A[AGlobIdx0];
        }
        F.endIf();
        F.beginIf(AIdx1 < APlaneSize);
        {
          AsTile[AsIdx1] = A[AGlobIdx1];
        }
        F.endIf();

        auto BPlaneSize = KTile * BlockTileN;
        auto BIdx0 = Tid * Two + Zero;
        auto BIdx1 = Tid * Two + One;
        auto BRow0 = BIdx0 / BlockTileN;
        auto BCol0 = BIdx0 % BlockTileN;
        auto BRow1 = BIdx1 / BlockTileN;
        auto BCol1 = BIdx1 % BlockTileN;
        auto BsIdx0 = BRow0 * BlockTileN + BCol0;
        auto BsIdx1 = BRow1 * BlockTileN + BCol1;
        auto BGlobIdx0 = (KBase + BRow0) * N + (BlockCol + BCol0);
        auto BGlobIdx1 = (KBase + BRow1) * N + (BlockCol + BCol1);
        F.beginIf(BIdx0 < BPlaneSize);
        {
          BsTile[BsIdx0] = B[BGlobIdx0];
        }
        F.endIf();
        F.beginIf(BIdx1 < BPlaneSize);
        {
          BsTile[BsIdx1] = B[BGlobIdx1];
        }
        F.endIf();

        F.callBuiltin(syncThreads);

        auto KIt = F.declVar<int>("KIt");
        F.forLoop<int>({KIt, Zero, KTile, One}, [&]() {
          auto I = F.declVar<int>("i");
          auto J = F.declVar<int>("j");

          F.forLoop<int>({I, Zero, RegTileM, One}, [&]() {
            auto r = Tidy * RegTileM + I;
            auto asIdx = r * KTile + KIt;
            Areg[I] = AsTile[asIdx];
          }).emit();

          F.forLoop<int>({J, Zero, RegTileN, One}, [&]() {
            auto c = Tidx * RegTileN + J;
            auto bsIdx = KIt * BlockTileN + c;
            Breg[J] = BsTile[bsIdx];
          }).emit();

          auto Ii = F.declVar<int>("ii");
          auto Jj = F.declVar<int>("jj");
          F.forLoop<int>({Ii, Zero, RegTileM, One}, [&]() {
            F.forLoop<int>({Jj, Zero, RegTileN, One}, [&]() {
              auto Cidx = Ii * RegTileN + Jj;
              Creg[Cidx] = Creg[Cidx] + (Areg[Ii] * Breg[Jj]);
            }).emit();
          }).emit();
        }).emit();

        F.callBuiltin(syncThreads);
      }).emit();

      {
        auto I = F.declVar<int>("i");
        auto J = F.declVar<int>("j");
        F.forLoop<int>({I, Zero, RegTileM, One}, [&]() {
          F.forLoop<int>({J, Zero, RegTileN, One}, [&]() {
            auto Cidx = (Row0 + I) * N + (Col0 + J);
            auto Ridx = I * RegTileN + J;
            C[Cidx] = Creg[Ridx];
          }).emit();
        }).emit();
      }

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

static bool verifyG(double *G, int N) {
  double expected = static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
  for (int i = 0; i < N * N; ++i) {
    if (G[i] != expected) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  proteus::init();

  constexpr int MatmulTileSize = 32;
  constexpr int RegTileM = 4;
  constexpr int RegTileN = 4;
  constexpr int BlockTileM = 64;
  constexpr int BlockTileN = 64;
  constexpr int KTile = 8;

  unsigned int N = 8192;
  int NumTrials = 5;
  bool DoVerify = true;
  std::string KernelType = "jit_regtiled";
  int blockTileMArg = BlockTileM;
  int blockTileNArg = BlockTileN;
  int kTileArg = KTile;
  int posIdx = 0;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--N") || !std::strcmp(argv[i], "-n")) {
      if (i + 1 < argc) N = static_cast<unsigned int>(std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) NumTrials = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--kernel")) {
      if (i + 1 < argc) {
        KernelType = argv[++i];
        if (KernelType != "jit" && KernelType != "jit_regtiled") {
          std::cerr << "Error: Invalid kernel type '" << KernelType
                    << "'. Valid options: jit, jit_regtiled\n";
          return 1;
        }
      }
    } else if (!std::strcmp(argv[i], "--verify")) {
      DoVerify = true;
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      DoVerify = false;
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [-n|--N N] [-t|--trials T] [--kernel KERNEL] [--verify|--no-verify]"
                << " [blockTileM blockTileN kTile]\n"
                << "  KERNEL: jit, jit_regtiled (default: jit_regtiled)\n"
                << "  Positional tile sizes are used for the JIT reg-tiled kernel;"
                << " defaults are " << BlockTileM << " " << BlockTileN << " " << KTile << "\n";
      return 0;
    } else {
      if (argv[i] && std::isdigit(static_cast<unsigned char>(argv[i][0]))) {
        int val = std::atoi(argv[i]);
        if (posIdx == 0) blockTileMArg = val;
        else if (posIdx == 1) blockTileNArg = val;
        else if (posIdx == 2) kTileArg = val;
        ++posIdx;
      }
    }
  }

  std::cout << "3mm PJ-DSL-GPU: N=" << N
            << ", trials=" << NumTrials
            << ", verify=" << DoVerify
            << ", kernel=" << KernelType
            << ", tiles=(blkM=" << blockTileMArg
            << ", blkN=" << blockTileNArg
            << ", k=" << kTileArg << ")" << std::endl;

  // Host allocations
  double *AH = (double *)new double[N * N];
  double *BH = (double *)new double[N * N];
  double *CH = (double *)new double[N * N];
  double *DH = (double *)new double[N * N];
  double *EH = (double *)new double[N * N];
  double *FH = (double *)new double[N * N];
  double *GH = (double *)new double[N * N];

  if (DoVerify) {
    for (unsigned int i = 0; i < N * N; ++i) {
      AH[i] = 1.0; BH[i] = 1.0; CH[i] = 1.0; DH[i] = 1.0;
      EH[i] = 0.0; FH[i] = 0.0; GH[i] = 0.0;
    }
  }

  // Device allocations
  double *AD, *BD, *CD, *DD, *ED, *FD, *GD;
  size_t Bytes = sizeof(double) * N * N;
  hipMalloc(reinterpret_cast<void **>(&AD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&BD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&CD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&DD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&ED), Bytes);
  hipMalloc(reinterpret_cast<void **>(&FD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&GD), Bytes);

  // Stage inputs
  hipMemcpy(AD, AH, Bytes, hipMemcpyHostToDevice);
  hipMemcpy(BD, BH, Bytes, hipMemcpyHostToDevice);
  hipMemcpy(CD, CH, Bytes, hipMemcpyHostToDevice);
  hipMemcpy(DD, DH, Bytes, hipMemcpyHostToDevice);


  if (KernelType == "jit_regtiled") {
    auto [JitMod, KernelHandle] = getRegSharedTiledMatmulKernel(N, blockTileMArg, blockTileNArg, kTileArg, RegTileM, RegTileN);
    JitMod->compile(true);

    unsigned int gridX = static_cast<unsigned int>(N / blockTileNArg);
    unsigned int gridY = static_cast<unsigned int>(N / blockTileMArg);
    unsigned int blockX = static_cast<unsigned int>(blockTileNArg / RegTileN);
    unsigned int blockY = static_cast<unsigned int>(blockTileMArg / RegTileM);

    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, ED, AD, BD);
    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, FD, CD, DD);
    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, GD, ED, FD);
    hipDeviceSynchronize();

    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NumTrials; ++t) {
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, ED, AD, BD);
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, FD, CD, DD);
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, GD, ED, FD);
    }
    hipDeviceSynchronize();

    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials (3 GEMMs): " << AvgMs << " ms" << '\n';

  } else {
    auto [JitMod, KernelHandle] = getMatmulKernel(N, MatmulTileSize);
    JitMod->compile(true);

    unsigned int gridX = static_cast<unsigned int>((N + MatmulTileSize - 1) / MatmulTileSize);
    unsigned int gridY = static_cast<unsigned int>((N + MatmulTileSize - 1) / MatmulTileSize);
    unsigned int blockX = static_cast<unsigned int>(MatmulTileSize);
    unsigned int blockY = static_cast<unsigned int>(MatmulTileSize);

    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, ED, AD, BD);
    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, FD, CD, DD);
    KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, GD, ED, FD);
    hipDeviceSynchronize();

    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NumTrials; ++t) {
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, ED, AD, BD);
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, FD, CD, DD);
      KernelHandle.launch({gridX, gridY, 1u}, {blockX, blockY, 1u}, 0, nullptr, GD, ED, FD);
    }
    hipDeviceSynchronize();

    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials (3 GEMMs): " << AvgMs << " ms" << '\n';
  }

  if (DoVerify) {
    hipMemcpy(GH, GD, Bytes, hipMemcpyDeviceToHost);
    if (verifyG(GH, N)) std::cout << "Verification passed" << std::endl;
    else                std::cout << "Verification failed" << std::endl;
  }

  hipFree(AD);
  hipFree(BD);
  hipFree(CD);
  hipFree(DD);
  hipFree(ED);
  hipFree(FD);
  hipFree(GD);

  delete[] AH; delete[] BH; delete[] CH; delete[] DH;
  delete[] EH; delete[] FH; delete[] GH;

  proteus::finalize();
  return 0;
}
