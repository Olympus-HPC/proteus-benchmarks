const char* const simulation_cu = "simulation.cu\n"
"#include \"rsbench.h\"\n"
"\n"
"////////////////////////////////////////////////////////////////////////////////////\n"
"// BASELINE FUNCTIONS\n"
"////////////////////////////////////////////////////////////////////////////////////\n"
"// All \"baseline\" code is at the top of this file. The baseline code is a simple\n"
"// implementation of the algorithm, with only minor CPU optimizations in place.\n"
"// Following these functions are a number of optimized variants,\n"
"// which each deploy a different combination of optimizations strategies. By\n"
"// default, RSBench will only run the baseline implementation. Optimized variants\n"
"// must be specifically selected using the \"-k <optimized variant ID>\" command\n"
"// line argument.\n"
"////////////////////////////////////////////////////////////////////////////////////\n"
"\n"
"template<\n"
"    int n_lookups,\n"
"    int input_doppler,\n"
"    int input_numL,\n"
"    int max_num_windows,\n"
"    int max_num_poles,\n"
"    int max_num_nucs\n"
">\n"
"__global__ \n"
"void lookup ( \n"
"    const int*__restrict__ num_nucs,\n"
"    const double*__restrict__ concs,\n"
"    const int*__restrict__ mats,\n"
"          int*__restrict__ verification,\n"
"    const int*__restrict__ n_windows,\n"
"    const double*__restrict__ pseudo_K0RS,\n"
"    const Window*__restrict__ windows,\n"
"    const Pole*__restrict__ poles) {\n"
"\n"
"  // get the index to operate on, first dimemsion\n"
"  size_t i = threadIdx.x + blockIdx.x * blockDim.x;\n"
"\n"
"  if (i < n_lookups) {\n"
"\n"
"    // Set the initial seed value\n"
"    uint64_t seed = STARTING_SEED;  \n"
"\n"
"    // Forward seed to lookup index (we need 2 samples per lookup)\n"
"    seed = fast_forward_LCG(seed, 2*i);\n"
"\n"
"    // Randomly pick an energy and material for the particle\n"
"    double p_energy = LCG_random_double(&seed);\n"
"    int mat         = pick_mat(&seed); \n"
"\n"
"    // debugging\n"
"    //printf(\"E = %lf mat = %d\\n\", p_energy, mat);\n"
"\n"
"    double macro_xs_vector[4] = {0};\n"
"\n"
"    // Perform macroscopic Cross Section Lookup\n"
"    calculate_macro_xs(\n"
"        macro_xs_vector, mat, p_energy, \n"
"        input_doppler, //in, \n"
"        input_numL,\n"
"        num_nucs, mats, \n"
"        max_num_nucs, concs, n_windows, pseudo_K0RS, windows, poles, \n"
"        max_num_windows, max_num_poles );\n"
"\n"
"    // For verification, and to prevent the compiler from optimizing\n"
"    // all work out, we interrogate the returned macro_xs_vector array\n"
"    // to find its maximum value index, then increment the verification\n"
"    // value by that index. In this implementation, we store to a global\n"
"    // array that will get tranferred back and reduced on the host.\n"
"    double max = -DBL_MAX;\n"
"    int max_idx = 0;\n"
"    for(int j = 0; j < 4; j++ )\n"
"    {\n"
"      if( macro_xs_vector[j] > max )\n"
"      {\n"
"        max = macro_xs_vector[j];\n"
"        max_idx = j;\n"
"      }\n"
"    }\n"
"    verification[i] = max_idx+1;\n"
"  }\n"
"}\n"
"\n"
"template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >\n"
"__device__\n"
"void calculate_macro_xs(double * macro_xs, int mat, double E,\n"
"                        int input_doppler, int input_numL,\n"
"                        INT_T num_nucs, INT_T mats,\n"
"                        int max_num_nucs,\n"
"                        DOUBLE_T concs,\n"
"                        INT_T n_windows,\n"
"                        DOUBLE_T pseudo_K0Rs,\n"
"                        WINDOW_T windows,\n"
"                        POLE_T poles,\n"
"                        int max_num_windows,\n"
"                        int max_num_poles ) \n"
"{\n"
"  // zero out macro vector\n"
"  for( int i = 0; i < 4; i++ )\n"
"    macro_xs[i] = 0;\n"
"\n"
"  // for nuclide in mat\n"
"  for( int i = 0; i < num_nucs[mat]; i++ )\n"
"  {\n"
"    double micro_xs[4];\n"
"    int nuc = mats[mat * max_num_nucs + i];\n"
"\n"
"    if( input_doppler == 1 )\n"
"      calculate_micro_xs_doppler( micro_xs, nuc, E, input_numL, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);\n"
"    else\n"
"      calculate_micro_xs( micro_xs, nuc, E, input_numL, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);\n"
"\n"
"    for( int j = 0; j < 4; j++ )\n"
"    {\n"
"      macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];\n"
"    }\n"
"    // Debug\n"
"    /*\n"
"       printf(\"E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\\n\",\n"
"       E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );\n"
"     */\n"
"  }\n"
"\n"
"  // Debug\n"
"  /*\n"
"     printf(\"E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\\n\",\n"
"     E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );\n"
"   */\n"
"}\n"
"\n"
"// No Temperature dependence (i.e., 0K evaluation)\n"
"template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >\n"
"__device__\n"
"void calculate_micro_xs(double * micro_xs, int nuc, double E, int input_numL,\n"
"                        INT_T n_windows, DOUBLE_T pseudo_K0RS, WINDOW_T windows,\n"
"                        POLE_T poles, int max_num_windows, int max_num_poles)\n"
"{\n"
"  // MicroScopic XS's to Calculate\n"
"  double sigT;\n"
"  double sigA;\n"
"  double sigF;\n"
"  double sigE;\n"
"\n"
"  // Calculate Window Index\n"
"  double spacing = 1.0 / n_windows[nuc];\n"
"  int window = (int) ( E / spacing );\n"
"  if( window == n_windows[nuc] )\n"
"    window--;\n"
"\n"
"  // Calculate sigTfactors\n"
"  RSComplex sigTfactors[4]; // Of length input.numL, which is always 4\n"
"  calculate_sig_T(nuc, E, input_numL, pseudo_K0RS, sigTfactors );\n"
"\n"
"  // Calculate contributions from window \"background\" (i.e., poles outside window (pre-calculated)\n"
"  Window w = windows[nuc * max_num_windows + window];\n"
"  sigT = E * w.T;\n"
"  sigA = E * w.A;\n"
"  sigF = E * w.F;\n"
"\n"
"  // Loop over Poles within window, add contributions\n"
"  for( int i = w.start; i < w.end; i++ )\n"
"  {\n"
"    RSComplex PSIIKI;\n"
"    RSComplex CDUM;\n"
"    Pole pole = poles[nuc * max_num_poles + i];\n"
"    RSComplex t1 = {0, 1};\n"
"    RSComplex t2 = {sqrt(E), 0 };\n"
"    PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );\n"
"    RSComplex E_c = {E, 0};\n"
"    CDUM = c_div(PSIIKI, E_c);\n"
"    sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;\n"
"    sigA += (c_mul( pole.MP_RA, CDUM)).r;\n"
"    sigF += (c_mul(pole.MP_RF, CDUM)).r;\n"
"  }\n"
"\n"
"  sigE = sigT - sigA;\n"
"\n"
"  micro_xs[0] = sigT;\n"
"  micro_xs[1] = sigA;\n"
"  micro_xs[2] = sigF;\n"
"  micro_xs[3] = sigE;\n"
"}\n"
"\n"
"// Temperature Dependent Variation of Kernel\n"
"// (This involves using the Complex Faddeeva function to\n"
"// Doppler broaden the poles within the window)\n"
"template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >\n"
"__device__\n"
"void calculate_micro_xs_doppler(double * micro_xs, int nuc, double E,\n"
"                                int input_numL, INT_T n_windows,\n"
"                                DOUBLE_T pseudo_K0RS, WINDOW_T windows,\n"
"                                POLE_T poles, int max_num_windows, int max_num_poles )\n"
"{\n"
"  // MicroScopic XS's to Calculate\n"
"  double sigT;\n"
"  double sigA;\n"
"  double sigF;\n"
"  double sigE;\n"
"\n"
"  // Calculate Window Index\n"
"  double spacing = 1.0 / n_windows[nuc];\n"
"  int window = (int) ( E / spacing );\n"
"  if( window == n_windows[nuc] )\n"
"    window--;\n"
"\n"
"  // Calculate sigTfactors\n"
"  RSComplex sigTfactors[4]; // Of length input.numL, which is always 4\n"
"  calculate_sig_T(nuc, E, input_numL, pseudo_K0RS, sigTfactors );\n"
"\n"
"  // Calculate contributions from window \"background\" (i.e., poles outside window (pre-calculated)\n"
"  Window w = windows[nuc * max_num_windows + window];\n"
"  sigT = E * w.T;\n"
"  sigA = E * w.A;\n"
"  sigF = E * w.F;\n"
"\n"
"  double dopp = 0.5;\n"
"\n"
"  // Loop over Poles within window, add contributions\n"
"  for( int i = w.start; i < w.end; i++ )\n"
"  {\n"
"    Pole pole = poles[nuc * max_num_poles + i];\n"
"\n"
"    // Prep Z\n"
"    RSComplex E_c = {E, 0};\n"
"    RSComplex dopp_c = {dopp, 0};\n"
"    RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);\n"
"\n"
"    // Evaluate Fadeeva Function\n"
"    RSComplex faddeeva = fast_nuclear_W( Z );\n"
"\n"
"    // Update W\n"
"    sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;\n"
"    sigA += (c_mul( pole.MP_RA , faddeeva)).r;\n"
"    sigF += (c_mul( pole.MP_RF , faddeeva)).r;\n"
"  }\n"
"\n"
"  sigE = sigT - sigA;\n"
"\n"
"  micro_xs[0] = sigT;\n"
"  micro_xs[1] = sigA;\n"
"  micro_xs[2] = sigF;\n"
"  micro_xs[3] = sigE;\n"
"}\n"
"\n"
"// picks a material based on a probabilistic distribution\n"
"__device__\n"
"int pick_mat( uint64_t * seed )\n"
"{\n"
"  // I have a nice spreadsheet supporting these numbers. They are\n"
"  // the fractions (by volume) of material in the core. Not a \n"
"  // *perfect* approximation of where XS lookups are going to occur,\n"
"  // but this will do a good job of biasing the system nonetheless.\n"
"\n"
"  double dist[12];\n"
"  dist[0]  = 0.140;  // fuel\n"
"  dist[1]  = 0.052;  // cladding\n"
"  dist[2]  = 0.275;  // cold, borated water\n"
"  dist[3]  = 0.134;  // hot, borated water\n"
"  dist[4]  = 0.154;  // RPV\n"
"  dist[5]  = 0.064;  // Lower, radial reflector\n"
"  dist[6]  = 0.066;  // Upper reflector / top plate\n"
"  dist[7]  = 0.055;  // bottom plate\n"
"  dist[8]  = 0.008;  // bottom nozzle\n"
"  dist[9]  = 0.015;  // top nozzle\n"
"  dist[10] = 0.025;  // top of fuel assemblies\n"
"  dist[11] = 0.013;  // bottom of fuel assemblies\n"
"\n"
"  double roll = LCG_random_double(seed);\n"
"\n"
"  // makes a pick based on the distro\n"
"  for( int i = 0; i < 12; i++ )\n"
"  {\n"
"    double running = 0;\n"
"    for( int j = i; j > 0; j-- )\n"
"      running += dist[j];\n"
"    if( roll < running )\n"
"      return i;\n"
"  }\n"
"\n"
"  return 0;\n"
"}\n"
"\n"
"template <class DOUBLE_T>\n"
"__device__\n"
"void calculate_sig_T( int nuc, double E, int input_numL, DOUBLE_T pseudo_K0RS, RSComplex * sigTfactors )\n"
"{\n"
"  double phi;\n"
"\n"
"  for( int i = 0; i < 4; i++ )\n"
"  {\n"
"    phi = pseudo_K0RS[nuc * input_numL + i] * sqrt(E);\n"
"\n"
"    if( i == 1 )\n"
"      phi -= - atan( phi );\n"
"    else if( i == 2 )\n"
"      phi -= atan( 3.0 * phi / (3.0 - phi*phi));\n"
"    else if( i == 3 )\n"
"      phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));\n"
"\n"
"    phi *= 2.0;\n"
"\n"
"    sigTfactors[i].r = cos(phi);\n"
"    sigTfactors[i].i = -sin(phi);\n"
"  }\n"
"}\n"
"\n"
"// This function uses a combination of the Abrarov Approximation\n"
"// and the QUICK_W three term asymptotic expansion.\n"
"// Only expected to use Abrarov ~0.5% of the time.\n"
"__device__\n"
"RSComplex fast_nuclear_W( RSComplex Z )\n"
"{\n"
"  // Abrarov \n"
"  if( c_abs(Z) < 6.0 )\n"
"  {\n"
"    // Precomputed parts for speeding things up\n"
"    // (N = 10, Tm = 12.0)\n"
"    RSComplex prefactor = {0, 8.124330e+01};\n"
"    double an[10] = {\n"
"      2.758402e-01,\n"
"      2.245740e-01,\n"
"      1.594149e-01,\n"
"      9.866577e-02,\n"
"      5.324414e-02,\n"
"      2.505215e-02,\n"
"      1.027747e-02,\n"
"      3.676164e-03,\n"
"      1.146494e-03,\n"
"      3.117570e-04\n"
"    };\n"
"    double neg_1n[10] = {\n"
"      -1.0,\n"
"      1.0,\n"
"      -1.0,\n"
"      1.0,\n"
"      -1.0,\n"
"      1.0,\n"
"      -1.0,\n"
"      1.0,\n"
"      -1.0,\n"
"      1.0\n"
"    };\n"
"\n"
"    double denominator_left[10] = {\n"
"      9.869604e+00,\n"
"      3.947842e+01,\n"
"      8.882644e+01,\n"
"      1.579137e+02,\n"
"      2.467401e+02,\n"
"      3.553058e+02,\n"
"      4.836106e+02,\n"
"      6.316547e+02,\n"
"      7.994380e+02,\n"
"      9.869604e+02\n"
"    };\n"
"\n"
"    RSComplex t1 = {0, 12};\n"
"    RSComplex t2 = {12, 0};\n"
"    RSComplex i = {0,1};\n"
"    RSComplex one = {1, 0};\n"
"    RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));\n"
"    RSComplex sum = {0,0};\n"
"    for( int n = 0; n < 10; n++ )\n"
"    {\n"
"      RSComplex t3 = {neg_1n[n], 0};\n"
"      RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);\n"
"      RSComplex t4 = {denominator_left[n], 0};\n"
"      RSComplex t5 = {144, 0};\n"
"      RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));\n"
"      RSComplex t6 = {an[n], 0};\n"
"      sum = c_add(sum, c_mul(t6, c_div(top,bot)));\n"
"    }\n"
"    W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));\n"
"    return W;\n"
"  }\n"
"  else\n"
"  {\n"
"    // QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).\n"
"    // Pre-computed parameters\n"
"    RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};\n"
"    RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};\n"
"    RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};\n"
"    RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};\n"
"\n"
"    RSComplex i = {0,1};\n"
"    RSComplex Z2 = c_mul(Z, Z);\n"
"    // Three Term Asymptotic Expansion\n"
"    RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));\n"
"\n"
"    return W;\n"
"  }\n"
"}\n"
"\n"
"__host__ __device__\n"
"double LCG_random_double(uint64_t * seed)\n"
"{\n"
"  const uint64_t m = 9223372036854775808ULL; // 2^63\n"
"  const uint64_t a = 2806196910506780709ULL;\n"
"  const uint64_t c = 1ULL;\n"
"  *seed = (a * (*seed) + c) % m;\n"
"  return (double) (*seed) / (double) m;\n"
"}  \n"
"\n"
"uint64_t LCG_random_int(uint64_t * seed)\n"
"{\n"
"  const uint64_t m = 9223372036854775808ULL; // 2^63\n"
"  const uint64_t a = 2806196910506780709ULL;\n"
"  const uint64_t c = 1ULL;\n"
"  *seed = (a * (*seed) + c) % m;\n"
"  return *seed;\n"
"}  \n"
"\n"
"__device__\n"
"uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)\n"
"{\n"
"  const uint64_t m = 9223372036854775808ULL; // 2^63\n"
"  uint64_t a = 2806196910506780709ULL;\n"
"  uint64_t c = 1ULL;\n"
"\n"
"  n = n % m;\n"
"\n"
"  uint64_t a_new = 1;\n"
"  uint64_t c_new = 0;\n"
"\n"
"  while(n > 0) \n"
"  {\n"
"    if(n & 1)\n"
"    {\n"
"      a_new *= a;\n"
"      c_new = c_new * a + c;\n"
"    }\n"
"    c *= (a + 1);\n"
"    a *= a;\n"
"\n"
"    n >>= 1;\n"
"  }\n"
"\n"
"  return (a_new * seed + c_new) % m;\n"
"}\n"
"\n"
"// Complex arithmetic functions\n"
"\n"
"__device__\n"
"RSComplex c_add( RSComplex A, RSComplex B)\n"
"{\n"
"  RSComplex C;\n"
"  C.r = A.r + B.r;\n"
"  C.i = A.i + B.i;\n"
"  return C;\n"
"}\n"
"\n"
"__device__\n"
"RSComplex c_sub( RSComplex A, RSComplex B)\n"
"{\n"
"  RSComplex C;\n"
"  C.r = A.r - B.r;\n"
"  C.i = A.i - B.i;\n"
"  return C;\n"
"}\n"
"\n"
"__host__ __device__\n"
"RSComplex c_mul( RSComplex A, RSComplex B)\n"
"{\n"
"  double a = A.r;\n"
"  double b = A.i;\n"
"  double c = B.r;\n"
"  double d = B.i;\n"
"  RSComplex C;\n"
"  C.r = (a*c) - (b*d);\n"
"  C.i = (a*d) + (b*c);\n"
"  return C;\n"
"}\n"
"\n"
"__device__\n"
"RSComplex c_div( RSComplex A, RSComplex B)\n"
"{\n"
"  double a = A.r;\n"
"  double b = A.i;\n"
"  double c = B.r;\n"
"  double d = B.i;\n"
"  RSComplex C;\n"
"  double denom = c*c + d*d;\n"
"  C.r = ( (a*c) + (b*d) ) / denom;\n"
"  C.i = ( (b*c) - (a*d) ) / denom;\n"
"  return C;\n"
"}\n"
"\n"
"__device__\n"
"double c_abs( RSComplex A)\n"
"{\n"
"  return sqrt(A.r*A.r + A.i * A.i);\n"
"}\n"
"\n"
"\n"
"// Fast (but inaccurate) exponential function\n"
"// Written By \"ACMer\":\n"
"// https://codingforspeed.com/using-faster-exponential-approximation/\n"
"// We use our own to avoid small differences in compiler specific\n"
"// exp() intrinsic implementations that make it difficult to verify\n"
"// if the code is working correctly or not.\n"
"__device__\n"
"double fast_exp(double x)\n"
"{\n"
"  x = 1.0 + x * 0.000244140625;\n"
"  x *= x; x *= x; x *= x; x *= x;\n"
"  x *= x; x *= x; x *= x; x *= x;\n"
"  x *= x; x *= x; x *= x; x *= x;\n"
"  return x;\n"
"}\n"
"\n"
"// Implementation based on:\n"
"// z = x + iy\n"
"// cexp(z) = e^x * (cos(y) + i * sin(y))\n"
"__device__\n"
"RSComplex fast_cexp( RSComplex z )\n"
"{\n"
"  double x = z.r;\n"
"  double y = z.i;\n"
"\n"
"  // For consistency across architectures, we\n"
"  // will use our own exponetial implementation\n"
"  //double t1 = exp(x);\n"
"  double t1 = fast_exp(x);\n"
"  double t2 = cos(y);\n"
"  double t3 = sin(y);\n"
"  RSComplex t4 = {t2, t3};\n"
"  RSComplex t5 = {t1, 0};\n"
"  RSComplex result = c_mul(t5, (t4));\n"
"  return result;\n"
"}  \n"
;
