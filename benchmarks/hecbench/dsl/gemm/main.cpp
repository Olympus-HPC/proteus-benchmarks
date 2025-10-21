#include <hip/hip_runtime.h>
#include <proteus/JitFrontend.hpp>
#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitInterface.hpp>

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <chrono>

using namespace proteus;
using namespace builtins::gpu;

#define TARGET "hip"




// clang-format off
// No FileCheck for now; program prints the resulting C matrix



static auto getMatmulKernel(int N, int TileSize) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<void(double *, double *, double *)>("tiled_matmul");
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
// 4x4 per-thread micro-tile, K tile = 8 (defaults; overridable via macros).

static auto getRegSharedTiledMatmulKernel(int N, int blockTileM, int blockTileN, int kTile, int regTileM, int regTileN) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<void(double *, double *, double *)>("reg_shared_tiled_matmul");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    // Shared tiles for current K-slice
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

      // Constants
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

      // Block origin in C
      auto BlockRow = Bidy * BlockTileM;
      auto BlockCol = Bidx * BlockTileN;

      // Per-thread micro tile origin in C
      auto Row0 = BlockRow + Tidy * RegTileM;
      auto Col0 = BlockCol + Tidx * RegTileN;

      // Zero accumulators
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

      // Loop over K dimension in tiles of kTile
      auto KBase = F.declVar<int>("KBase");
      F.forLoop<int>({KBase, Zero, Nvar, KTile}, [&]() {
        // Cooperative load of A and B tiles into shared memory.
        auto Tid = Tidy * ThreadsX + Tidx; // 0 .. (blockTileM/regTileM*blockTileN/regTileN - 1)

        // Load A tile: size [blockTileM x kTile]
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

        // Load B tile: size [kTile x blockTileN]
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

        // Compute this micro-tile using the shared tiles and register blocking
        auto KIt = F.declVar<int>("KIt");
        F.forLoop<int>({KIt, Zero, KTile, One}, [&]() {
          // Load rows/cols into registers
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

          // FMA on the micro-tile
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

      // Write back the per-thread micro-tile to C
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

bool verify(double *C, int N) {
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if (C[I*N + J] != N) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  proteus::init();

  // Default tile size constants
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
      if (i + 1 < argc) {
        N = static_cast<unsigned int>(std::atoi(argv[++i]));
      }
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) {
        NumTrials = std::atoi(argv[++i]);
      }
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
      // Positional ints for blockTileM, blockTileN, kTile (for JIT reg-tiled)
      if (argv[i] && std::isdigit(static_cast<unsigned char>(argv[i][0]))) {
        int val = std::atoi(argv[i]);
        if (posIdx == 0) blockTileMArg = val;
        else if (posIdx == 1) blockTileNArg = val;
        else if (posIdx == 2) kTileArg = val;
        ++posIdx;
      }
    }
  }
  std::cout << "Configuration: (N, NumTrials, DoVerify, Kernel, Tiles) = ("
            << N << ", " << NumTrials << ", " << DoVerify << ", " << KernelType
            << ", blkM=" << blockTileMArg << ", blkN=" << blockTileNArg
            << ", k=" << kTileArg << ")" << std::endl;

  // Host allocations
  double *AH = (double *)new double[N * N];
  double *BH = (double *)new double[N * N];
  double *CH = (double *)new double[N * N];

  // Device allocations
  double *AD;
  double *BD;
  double *CD;
  size_t Bytes = sizeof(double) * N * N;
  hipMalloc(reinterpret_cast<void **>(&AD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&BD), Bytes);
  hipMalloc(reinterpret_cast<void **>(&CD), Bytes);

  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      AH[I * N + J] = 1.0;
      BH[I * N + J] = 1.0;
      CH[I * N + J] = 0.0;
    }
  }
  // Stage inputs to device
  hipMemcpy(AD, AH, Bytes, hipMemcpyHostToDevice);
  hipMemcpy(BD, BH, Bytes, hipMemcpyHostToDevice);
  hipMemcpy(CD, CH, Bytes, hipMemcpyHostToDevice);

  // Kernel execution based on type
  if (KernelType == "jit_regtiled") {
    auto [JitMod, KernelHandle] = getRegSharedTiledMatmulKernel(N, blockTileMArg, blockTileNArg, kTileArg, RegTileM, RegTileN);
    JitMod->compile(true);
    KernelHandle.launch({static_cast<unsigned int>(N / blockTileNArg), static_cast<unsigned int>(N / blockTileMArg), 1u}, {static_cast<unsigned int>(blockTileNArg / RegTileN), static_cast<unsigned int>(blockTileMArg / RegTileM), 1u}, 0, nullptr, CD, AD, BD);
    hipDeviceSynchronize();

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      KernelHandle.launch({static_cast<unsigned int>(N / blockTileNArg), static_cast<unsigned int>(N / blockTileMArg), 1u}, {static_cast<unsigned int>(blockTileNArg / RegTileN), static_cast<unsigned int>(blockTileMArg / RegTileM), 1u}, 0, nullptr, CD, AD, BD);

    }
    hipDeviceSynchronize();
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else if (KernelType == "jit") {
    auto [JitMod, KernelHandle] = getMatmulKernel(N, MatmulTileSize);
    JitMod->compile(true);
    KernelHandle.launch({(N + MatmulTileSize - 1) / MatmulTileSize, (N + MatmulTileSize - 1) / MatmulTileSize, 1}, {MatmulTileSize, MatmulTileSize, 1}, 0, nullptr, CD, AD, BD);
    hipDeviceSynchronize();

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      KernelHandle.launch({(N + MatmulTileSize - 1) / MatmulTileSize, (N + MatmulTileSize - 1) / MatmulTileSize, 1}, {MatmulTileSize, MatmulTileSize, 1}, 0, nullptr, CD, AD, BD);
    }
    hipDeviceSynchronize();
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';
  }

  // Final result back to host after timing loop
  if (DoVerify) {
    hipMemcpy(CH, CD, Bytes, hipMemcpyDeviceToHost);
    verify(CH, N);
    std::cout << "Verification passed" << std::endl;
  }

  // Cleanup device and host memory
  hipFree(AD);
  hipFree(BD);
  hipFree(CD);
  delete[] AH;
  delete[] BH;
  delete[] CH;

  proteus::finalize();
  return 0;

}
