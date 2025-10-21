#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <proteus/JitFrontend.hpp>
#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitInterface.hpp>

#include "bude.h"

using namespace proteus;
using namespace builtins::gpu;

#define TARGET "hip"

namespace {

typedef std::chrono::high_resolution_clock::time_point TimePoint;

struct Params {
  size_t natlig;
  size_t natpro;
  size_t ntypes;
  size_t nposes;

  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  size_t iterations;

  size_t posesPerWI;
  size_t wgSize;
  std::string deckDir;

  friend std::ostream &operator<<(std::ostream &os, const Params &params) {
    os << "natlig:      " << params.natlig << "\n"
       << "natpro:      " << params.natpro << "\n"
       << "ntypes:      " << params.ntypes << "\n"
       << "nposes:      " << params.nposes << "\n"
       << "iterations:  " << params.iterations << "\n"
       << "posesPerWI:  " << params.posesPerWI << "\n"
       << "wgSize:      " << params.wgSize << "\n";
    return os;
  }
};

static auto buildFastenKernel(size_t posesPerWI_, size_t ntypes_, size_t nposes_, size_t natlig_, size_t natpro_) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle = JitMod->addKernel<void(float*,float*,float*,int32_t*,float*,float*,float*,int32_t*,float*,float*,float*,float*,float*,float*,int32_t*, float*, float*, float*, float*)>("fasten_main");
  auto &F = KernelHandle.F;
  F.beginFunction();
  {
    auto [protein_x, protein_y, protein_z, protein_type,
          ligand_x, ligand_y, ligand_z, ligand_type,
          transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5,
          ff_hbtype, ff_radius, ff_hphb, ff_elsc, etotals] = F.getArgs();

    auto lid = F.callBuiltin(getThreadIdX);
    auto gid = F.callBuiltin(getBlockIdX);
    auto lrange = F.callBuiltin(getBlockDimX);
    auto ZERO = F.defRuntimeConst<float>(0.0f);
    auto QUARTER = F.defRuntimeConst<float>(0.25f);
    auto HALF = F.defRuntimeConst<float>(0.5f);
    auto ONE = F.defRuntimeConst<float>(1.0f);
    auto TWO = F.defRuntimeConst<float>(2.0f);
    auto FOUR = F.defRuntimeConst<float>(4.0f);
    auto CNSTNT = F.defRuntimeConst<float>(45.0f);

    // Energy evaluation parameters
    auto HBTYPE_F = F.defRuntimeConst<int>(70);
    auto HBTYPE_E = F.defRuntimeConst<int>(69);
    auto HARDNESS = F.defRuntimeConst<float>(38.0f);
    auto NPNPDIST = F.defRuntimeConst<float>(5.5f);
    auto NPPDIST = F.defRuntimeConst<float>(1.0f);

    auto FLT_MAX = F.defRuntimeConst<float>(std::numeric_limits<float>::max());

    auto [posesPerWI, ntypes, nposes, natlig, natpro] = F.defRuntimeConsts(posesPerWI_, ntypes_, nposes_, natlig_, natpro_);

    auto etot = F.declVar<float[]>(posesPerWI_);

    // Positions
    auto lpos_x = F.declVar<float[]>(posesPerWI_);
    auto lpos_y = F.declVar<float[]>(posesPerWI_);
    auto lpos_z = F.declVar<float[]>(posesPerWI_);

    // Transformations
    auto transform_x = F.declVar<float[]>(posesPerWI_ * 3);
    auto transform_y = F.declVar<float[]>(posesPerWI_ * 3);
    auto transform_z = F.declVar<float[]>(posesPerWI_ * 3);
    auto transform_w = F.declVar<float[]>(posesPerWI_ * 3);

    auto ix = gid * lrange * posesPerWI + lid;

    F.beginIf(ix >= nposes);
    { ix = nposes - posesPerWI; }
    F.endIf();

    auto i = F.defVar<size_t>(0);
    F.beginFor(i, i, posesPerWI, F.defRuntimeConst<size_t>(1));
    {
      auto index = ix + i * lrange;

      auto sx = sinf(transforms_0[index]);
      auto cx = cosf(transforms_0[index]);
      auto sy = sinf(transforms_1[index]);
      auto cy = cosf(transforms_1[index]);
      auto sz = sinf(transforms_2[index]);
      auto cz = cosf(transforms_2[index]);

      auto base = i * 3;
      transform_x[base + 0] = cy * cz;
      transform_y[base + 0] = sx * sy * cz - cx * sz;
      transform_z[base + 0] = cx * sy * cz + sx * sz;
      transform_w[base + 0] = transforms_3[index];

      transform_x[base + 1] = cy * sz;
      transform_y[base + 1] = sx * sy * sz + cx * cz;
      transform_z[base + 1] = cx * sy * sz - sx * cz;
      transform_w[base + 1] = transforms_4[index];

      transform_x[base + 2] = -1.0f * sy;
      transform_y[base + 2] = sx * cy;
      transform_z[base + 2] = cx * cy;
      transform_w[base + 2] = transforms_5[index];

      etot[i] = ZERO;
    }
    F.endFor();


    auto il = F.defVar<size_t>(0);
    // Loop over ligand atoms
    F.beginFor(il, il, natlig, F.defRuntimeConst<size_t>(1));
    {
      // Load ligand atom data
      auto l_t = ligand_type[il];
      auto l_x = ligand_x[il];
      auto l_y = ligand_y[il];
      auto l_z = ligand_z[il];

      auto l_hbtype = ff_hbtype[l_t];
      auto l_radius = ff_radius[l_t];
      auto l_hphb = ff_hphb[l_t];
      auto l_elsc = ff_elsc[l_t];

      auto lhphb_ltz = l_hphb < ZERO;
      auto lhphb_gtz = l_hphb > ZERO;

      i = 0;
      F.beginFor(i, i, posesPerWI, F.defRuntimeConst<size_t>(1));
      {
        auto base = i * 3;
        lpos_x[i] = transform_w[base + 0] + l_x * transform_x[base + 0] + l_y * transform_y[base + 0] + l_z * transform_z[base + 0];
        lpos_y[i] = transform_w[base + 1] + l_x * transform_x[base + 1] + l_y * transform_y[base + 1] + l_z * transform_z[base + 1];
        lpos_z[i] = transform_w[base + 2] + l_x * transform_x[base + 2] + l_y * transform_y[base + 2] + l_z * transform_z[base + 2];
      }
      F.endFor();
      auto ip = F.defVar<size_t>(0);
      // Loop over protein atoms
      F.beginFor(ip, ip, natpro, F.defRuntimeConst<size_t>(1));
      {
        // Load protein atom data
        auto p_t = protein_type[ip];
        auto p_x = protein_x[ip];
        auto p_y = protein_y[ip];
        auto p_z = protein_z[ip];

        auto p_hbtype = ff_hbtype[p_t];
        auto p_radius = ff_radius[p_t];
        auto p_hphb = ff_hphb[p_t];
        auto p_elsc = ff_elsc[p_t];

        auto radij = p_radius + l_radius;
        auto r_radij = 1.f / (radij);

        auto elcdst = F.defVar<float>(TWO);
        F.beginIf(p_hbtype == HBTYPE_F);
        {
          F.beginIf(l_hbtype == HBTYPE_F);
          { elcdst = FOUR; }
          F.endIf();
        }
        F.endIf();
        auto elcdst1 = F.defVar<float>(HALF);
        F.beginIf(p_hbtype == HBTYPE_F);
        {
          F.beginIf(l_hbtype == HBTYPE_F);
          { elcdst1 = QUARTER; }
          F.endIf();
        }
        F.endIf();

        auto type_E = F.defVar<bool>(false);
        F.beginIf(p_hbtype == HBTYPE_E);
        { type_E = true; }
        F.endIf();
        F.beginIf(l_hbtype == HBTYPE_E);
        { type_E = true; }
        F.endIf();

        auto phphb_ltz = p_hphb < ZERO;
        auto phphb_gtz = p_hphb > ZERO;
        auto phphb_nz = p_hphb != ZERO;

        auto p_hphb_eff = p_hphb * ONE;
        F.beginIf(phphb_ltz);
        {
          F.beginIf(lhphb_gtz);
          { p_hphb_eff = p_hphb * (-1.0f * ONE); }
          F.endIf();
        }
        F.endIf();
        auto l_hphb_eff = l_hphb * ONE;
        F.beginIf(phphb_gtz);
        {
          F.beginIf(lhphb_ltz);
          { l_hphb_eff = l_hphb * (-1.0f * ONE); }
          F.endIf();
        }
        F.endIf();

        auto distdslv = -1.0f * FLT_MAX;
        F.beginIf(phphb_ltz);
        {
          distdslv = NPPDIST;
          F.beginIf(lhphb_ltz);
          { distdslv = NPNPDIST; }
          F.endIf();
        }
        F.endIf();
        F.beginIf(phphb_ltz == false);
        {
          F.beginIf(lhphb_ltz);
          { distdslv = NPPDIST; }
          F.endIf();
        }
        F.endIf();
        auto r_distdslv = 1.f / (distdslv);

        auto chrg_init = l_elsc * p_elsc;
        auto dslv_init = p_hphb_eff + l_hphb_eff;

        i = 0;
        F.beginFor(i, i, posesPerWI, F.defRuntimeConst<size_t>(1));
        {
          auto x = lpos_x[i] - p_x;
          auto y = lpos_y[i] - p_y;
          auto z = lpos_z[i] - p_z;

          auto distij = sqrtf(x * x + y * y + z * z);

          // Calculate the sum of the sphere radii
          auto distbb = distij - radij;
          auto zone1 = (distbb < ZERO);

          // Calculate steric energy
          auto steric = ONE - (distij * r_radij);
          auto multiplier = F.defVar<float>(ZERO);
          F.beginIf(zone1);
          {
            multiplier = TWO * HARDNESS;
          }
          F.endIf();
          etot[i] += steric * multiplier;

          // Calculate formal and dipole charge interactions
          auto chrg_zone_factor = ONE - distbb * elcdst1;
          F.beginIf(zone1);
          { chrg_zone_factor = ONE; }
          F.endIf();
          auto chrg_dist_factor = F.defVar<float>(ZERO);
          F.beginIf(distbb < elcdst);
          { chrg_dist_factor = ONE; }
          F.endIf();
          auto chrg_e = chrg_init * chrg_zone_factor * chrg_dist_factor;
          auto neg_chrg_e = -1.0f * fabs(chrg_e);
          F.beginIf(type_E);
          { chrg_e = neg_chrg_e; }
          F.endIf();
          etot[i] += chrg_e * CNSTNT;

          // Calculate the two cases for Nonpolar-Polar repulsive interactions
          auto coeff = ONE - (distbb * r_distdslv);
          auto dslv_zone_factor = F.defVar<float>(ZERO);
          F.beginIf(distbb < distdslv);
          {
            F.beginIf(phphb_nz);
            { dslv_zone_factor = ONE; }
            F.endIf();
          }
          F.endIf();
          auto dslv_e = dslv_init * dslv_zone_factor;
          auto dslv_scale = coeff;
          F.beginIf(zone1);
          { dslv_scale = ONE; }
          F.endIf();
          dslv_e *= dslv_scale;
          etot[i] += dslv_e;

        }
        F.endFor();
    }
    F.endFor();
  }
  F.endFor();


  auto td_base = gid * lrange * posesPerWI + lid;
  F.beginIf(td_base < nposes);
  {
      i = 0;
      F.beginFor(i, i, posesPerWI, F.defRuntimeConst<size_t>(1));
      {
      etotals[td_base + i * lrange] = etot[i] * HALF;
      }
      F.endFor();
  }
  F.endIf();
  F.ret();
  }
  F.endFunction();

  std::cout << "Kernel constructed successfully" << std::endl;
  return std::make_pair(std::move(JitMod), KernelHandle);
}

double elapsedMillis(const TimePoint &start, const TimePoint &end) {
  auto elapsedNs =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              end - start)
                              .count());
  return elapsedNs * 1e-6;
}

void printTimings(const Params &params, double millis) {
  double ms = (millis / params.iterations);
  double runtime = ms * 1e-3;

  double ops_per_wg =
      params.posesPerWI * 27 +
      params.natlig * (3 + params.posesPerWI * 18 +
                       params.natpro * (11 + params.posesPerWI * 30)) +
      params.posesPerWI;
  double total_ops = ops_per_wg * (static_cast<double>(params.nposes) /
                                   params.posesPerWI);
  double flops = total_ops / runtime;
  double gflops = flops / 1e9;

  double interactions = static_cast<double>(params.nposes) *
                        static_cast<double>(params.natlig) *
                        static_cast<double>(params.natpro);
  double interactions_per_sec = interactions / runtime;

  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "- Total kernel time:    " << (millis) << " ms\n";
  std::cout << "- Average kernel time:   " << ms << " ms\n";
  std::cout << "- Interactions/s: " << (interactions_per_sec / 1e9)
            << " billion\n";
  std::cout << "- GFLOP/s:        " << gflops << "\n";
}

template <typename T> std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<T> xs(static_cast<size_t>(len) / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

Params loadParameters(const std::vector<std::string> &args) {
  Params params = {};

  params.iterations = DEFAULT_ITERS;
  params.nposes = DEFAULT_NPOSES;
  params.wgSize = DEFAULT_WGSIZE;
  params.deckDir = DATA_DIR;
  params.posesPerWI = DEFAULT_PPWI;

  const auto readParam = [&args](size_t &current, const std::string &arg,
                                 const std::initializer_list<std::string> &matches,
                                 const std::function<void(std::string)> &handle) {
    if (matches.size() == 0)
      return false;
    if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
      if (current + 1 < args.size()) {
        current++;
        handle(args[current]);
      } else {
        std::cerr << "[";
        for (const auto &m : matches)
          std::cerr << m;
        std::cerr << "] specified but no value was given" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return true;
    }
    return false;
  };

  const auto bindInt = [](const std::string &param, size_t &dest,
                          const std::string &name) {
    try {
      auto parsed = std::stol(param);
      if (parsed < 0) {
        std::cerr << "positive integer required for <" << name
                  << ">: `" << parsed << "`" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      dest = static_cast<size_t>(parsed);
    } catch (...) {
      std::cerr << "malformed value, integer required for <" << name
                << ">: `" << param << "`" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  for (size_t i = 0; i < args.size(); ++i) {
    using namespace std::placeholders;
    const auto arg = args[i];
    if (readParam(i, arg, {"--iterations", "-i"},
                  std::bind(bindInt, _1, std::ref(params.iterations),
                            "iterations")))
      continue;
    if (readParam(i, arg, {"--numposes", "-n"},
                  std::bind(bindInt, _1, std::ref(params.nposes), "numposes")))
      continue;
    if (readParam(i, arg, {"--posesperwi", "-p"},
                  std::bind(bindInt, _1, std::ref(params.posesPerWI),
                            "posesperwi")))
      continue;
    if (readParam(i, arg, {"--wgsize", "-w"},
                  std::bind(bindInt, _1, std::ref(params.wgSize), "wgsize")))
      continue;
    if (readParam(i, arg, {"--deck"},
                  [&](const std::string &param) { params.deckDir = param; }))
      continue;

    if (arg == "--help" || arg == "-h") {
      std::cout << "\n";
      std::cout << "Usage: ./main [OPTIONS]\n\n"
                << "Options:\n"
                << "  -h  --help               Print this message\n"
                << "  -i  --iterations I       Repeat kernel I times (default: "
                << DEFAULT_ITERS << ")\n"
                << "  -n  --numposes   N       Compute energies for N poses (default: "
                << DEFAULT_NPOSES << ")\n"
                << "  -p  --posesperwi PPWI    Compute PPWI poses per work-item "
                   "(default: "
                << DEFAULT_PPWI << ")\n"
                << "  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE using "
                   "nd_range, set to 0 for plain range (default: "
                << DEFAULT_WGSIZE << ")\n"
                << "      --deck       DECK    Use the DECK directory as input deck "
                   "(default: "
                << DATA_DIR << ")"
                << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    std::cout << "Unrecognized argument '" << arg << "' (try '--help')"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (params.posesPerWI == 0 || params.posesPerWI > MAX_PPWI) {
    std::cerr << "posesperwi must be in 1.." << MAX_PPWI << std::endl;
    std::exit(EXIT_FAILURE);
  }

  params.ligand = readNStruct<Atom>(params.deckDir + FILE_LIGAND);
  params.natlig = params.ligand.size();

  params.protein = readNStruct<Atom>(params.deckDir + FILE_PROTEIN);
  params.natpro = params.protein.size();

  params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);
  params.ntypes = params.forcefield.size();

  auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
  if (poses.size() / 6 != params.nposes) {
    throw std::invalid_argument("Bad poses: " +
                                std::to_string(poses.size()));
  }

  for (size_t i = 0; i < 6; ++i) {
    params.poses[i].resize(params.nposes);
    std::copy(std::next(poses.cbegin(), i * params.nposes),
              std::next(poses.cbegin(), i * params.nposes + params.nposes),
              params.poses[i].begin());
  }

  return params;
}

std::vector<float> runKernel(const Params &params) {
  auto energies = std::vector<float>(params.nposes);

  std::vector<float> h_protein_x(params.natpro), h_protein_y(params.natpro),
      h_protein_z(params.natpro);
  std::vector<int32_t> h_protein_type(params.natpro);
  for (size_t i = 0; i < params.natpro; ++i) {
    h_protein_x[i] = params.protein[i].x;
    h_protein_y[i] = params.protein[i].y;
    h_protein_z[i] = params.protein[i].z;
    h_protein_type[i] = params.protein[i].type;
  }

  std::vector<float> h_ligand_x(params.natlig), h_ligand_y(params.natlig),
      h_ligand_z(params.natlig);
  std::vector<int32_t> h_ligand_type(params.natlig);
  for (size_t i = 0; i < params.natlig; ++i) {
    h_ligand_x[i] = params.ligand[i].x;
    h_ligand_y[i] = params.ligand[i].y;
    h_ligand_z[i] = params.ligand[i].z;
    h_ligand_type[i] = params.ligand[i].type;
  }

  float *d_protein_x = nullptr;
  float *d_protein_y = nullptr;
  float *d_protein_z = nullptr;
  int32_t *d_protein_type = nullptr;
  hipMalloc(reinterpret_cast<void **>(&d_protein_x),
            params.natpro * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_protein_y),
            params.natpro * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_protein_z),
            params.natpro * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_protein_type),
            params.natpro * sizeof(int32_t));
  hipMemcpy(d_protein_x, h_protein_x.data(),
            params.natpro * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_protein_y, h_protein_y.data(),
            params.natpro * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_protein_z, h_protein_z.data(),
            params.natpro * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_protein_type, h_protein_type.data(),
            params.natpro * sizeof(int32_t),
            hipMemcpyHostToDevice);

  float *d_ligand_x = nullptr;
  float *d_ligand_y = nullptr;
  float *d_ligand_z = nullptr;
  int32_t *d_ligand_type = nullptr;
  hipMalloc(reinterpret_cast<void **>(&d_ligand_x),
            params.natlig * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_ligand_y),
            params.natlig * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_ligand_z),
            params.natlig * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_ligand_type),
            params.natlig * sizeof(int32_t));
  hipMemcpy(d_ligand_x, h_ligand_x.data(),
            params.natlig * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_y, h_ligand_y.data(),
            params.natlig * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_z, h_ligand_z.data(),
            params.natlig * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ligand_type, h_ligand_type.data(),
            params.natlig * sizeof(int32_t),
            hipMemcpyHostToDevice);

  float *transforms_0 = nullptr;
  float *transforms_1 = nullptr;
  float *transforms_2 = nullptr;
  float *transforms_3 = nullptr;
  float *transforms_4 = nullptr;
  float *transforms_5 = nullptr;
  hipMalloc(reinterpret_cast<void **>(&transforms_0),
            params.nposes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&transforms_1),
            params.nposes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&transforms_2),
            params.nposes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&transforms_3),
            params.nposes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&transforms_4),
            params.nposes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&transforms_5),
            params.nposes * sizeof(float));

  float *transformPtrs[6] = {transforms_0, transforms_1, transforms_2,
                             transforms_3, transforms_4, transforms_5};
  for (size_t i = 0; i < 6; ++i) {
    hipMemcpy(transformPtrs[i], params.poses[i].data(),
              params.nposes * sizeof(float),
              hipMemcpyHostToDevice);
  }

  std::vector<int32_t> h_ff_hbtype(params.ntypes);
  std::vector<float> h_ff_radius(params.ntypes), h_ff_hphb(params.ntypes),
      h_ff_elsc(params.ntypes);
  for (size_t i = 0; i < params.ntypes; ++i) {
    h_ff_hbtype[i] = params.forcefield[i].hbtype;
    h_ff_radius[i] = params.forcefield[i].radius;
    h_ff_hphb[i] = params.forcefield[i].hphb;
    h_ff_elsc[i] = params.forcefield[i].elsc;
  }

  int32_t *d_ff_hbtype = nullptr;
  float *d_ff_radius = nullptr;
  float *d_ff_hphb = nullptr;
  float *d_ff_elsc = nullptr;
  hipMalloc(reinterpret_cast<void **>(&d_ff_hbtype),
            params.ntypes * sizeof(int32_t));
  hipMalloc(reinterpret_cast<void **>(&d_ff_radius),
            params.ntypes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_ff_hphb),
            params.ntypes * sizeof(float));
  hipMalloc(reinterpret_cast<void **>(&d_ff_elsc),
            params.ntypes * sizeof(float));
  hipMemcpy(d_ff_hbtype, h_ff_hbtype.data(),
            params.ntypes * sizeof(int32_t),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ff_radius, h_ff_radius.data(),
            params.ntypes * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ff_hphb, h_ff_hphb.data(),
            params.ntypes * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_ff_elsc, h_ff_elsc.data(),
            params.ntypes * sizeof(float),
            hipMemcpyHostToDevice);

  float *results = nullptr;
  hipMalloc(reinterpret_cast<void **>(&results),
            params.nposes * sizeof(float));

  auto [JitMod, KernelHandle] = buildFastenKernel(params.posesPerWI, params.ntypes, params.nposes, params.natlig, params.natpro);
  JitMod->compile(true);
  // JitMod->print();

  double global = std::ceil(static_cast<double>(params.nposes) /
                            static_cast<double>(params.posesPerWI));
  global = std::ceil(global / static_cast<double>(params.wgSize));

  const unsigned int gridDimX = static_cast<unsigned int>(global);
  const unsigned int blockDimX = static_cast<unsigned int>(params.wgSize);


  KernelHandle.launch({gridDimX, 1u, 1u},
                      {blockDimX, 1u, 1u},
                      0, nullptr,
                      d_protein_x, d_protein_y,
                      d_protein_z, d_protein_type, d_ligand_x,
                      d_ligand_y, d_ligand_z, d_ligand_type,
                      transforms_0, transforms_1, transforms_2,
                      transforms_3, transforms_4, transforms_5,
                      d_ff_hbtype, d_ff_radius, d_ff_hphb,
                      d_ff_elsc, results);
  hipDeviceSynchronize();

  auto kernelStart = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < params.iterations; ++i) {
    KernelHandle.launch({gridDimX, 1u, 1u},
                        {blockDimX, 1u, 1u},
                        0, nullptr,
                        d_protein_x, d_protein_y,
                        d_protein_z, d_protein_type, d_ligand_x,
                        d_ligand_y, d_ligand_z, d_ligand_type,
                        transforms_0, transforms_1, transforms_2,
                        transforms_3, transforms_4, transforms_5,
                        d_ff_hbtype, d_ff_radius, d_ff_hphb,
                        d_ff_elsc, results);

  }
  hipDeviceSynchronize();
  auto kernelEnd = std::chrono::high_resolution_clock::now();

  hipMemcpy(energies.data(), results,
            params.nposes * sizeof(float),
            hipMemcpyDeviceToHost);

  printTimings(params, elapsedMillis(kernelStart, kernelEnd));

  hipFree(d_protein_x);
  hipFree(d_protein_y);
  hipFree(d_protein_z);
  hipFree(d_protein_type);
  hipFree(d_ligand_x);
  hipFree(d_ligand_y);
  hipFree(d_ligand_z);
  hipFree(d_ligand_type);
  hipFree(transforms_0);
  hipFree(transforms_1);
  hipFree(transforms_2);
  hipFree(transforms_3);
  hipFree(transforms_4);
  hipFree(transforms_5);
  hipFree(d_ff_hbtype);
  hipFree(d_ff_radius);
  hipFree(d_ff_hphb);
  hipFree(d_ff_elsc);
  hipFree(results);

  return energies;
}

} // namespace

int main(int argc, char *argv[]) {
  proteus::init();

  auto args = std::vector<std::string>(argv + 1, argv + argc);
  auto params = loadParameters(args);

  std::cout << "Poses     : " << params.nposes << std::endl;
  std::cout << "Iterations: " << params.iterations << std::endl;
  std::cout << "Ligands   : " << params.natlig << std::endl;
  std::cout << "Proteins  : " << params.natpro << std::endl;
  std::cout << "Deck      : " << params.deckDir << std::endl;
  std::cout << "WG        : " << params.wgSize << std::endl;
  auto energies = runKernel(params);

#ifdef DUMP
  FILE *output = fopen("result.out", "w+");
  printf("\nEnergies\n");
  for (size_t i = 0; i < params.nposes; i++) {
    fprintf(output, "%7.2f\n", energies[i]);
    if (i < 16)
      printf("%7.2f\n", energies[i]);
  }
  fclose(output);
#endif

  std::ifstream refEnergies(params.deckDir + FILE_REF_ENERGIES);
  size_t nRefPoses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    std::cout << "Only validating the first " << REF_NPOSES << " poses.\n";
    nRefPoses = REF_NPOSES;
  }

  std::string line;
  float maxdiff = 0.0f;
  for (size_t i = 0; i < nRefPoses; i++) {
    if (!std::getline(refEnergies, line)) {
      throw std::logic_error("ran out of ref energies lines to verify");
    }
    float e = std::stof(line);
    if (std::fabs(e) < 1.f && std::fabs(energies[i]) < 1.f)
      continue;

    float diff = std::fabs(e - energies[i]) / e;
    if (diff > maxdiff)
      maxdiff = diff;
  }
  std::cout << "Largest difference was "
            << std::setprecision(3) << (100 * maxdiff) << "%.\n\n";

  return 0;
}
