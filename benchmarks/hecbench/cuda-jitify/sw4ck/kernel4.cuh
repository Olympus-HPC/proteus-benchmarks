const char* const kernel4_cpp =
"template<\n"
"    const int start0, const int N0, \n"
"    const int start1, const int N1,\n"
"    const int start2, const int N2,\n"
"    const int ifirst, const int ilast,\n"
"    const int jfirst, const int jlast,\n"
"    const int kfirst, const int klast,\n"
"    const float_sw4 a1, const float_sw4 sgn\n"
">\n"
"__global__ \n"
"void kernel4(\n"
"    const float_sw4* __restrict__ a_u, \n"
"    const float_sw4* __restrict__ a_mu,\n"
"    const float_sw4* __restrict__ a_lambda,\n"
"    const float_sw4* __restrict__ a_met,\n"
"    const float_sw4* __restrict__ a_jac,\n"
"          float_sw4* __restrict__ a_lu, \n"
"    const float_sw4* __restrict__ a_acof, \n"
"    const float_sw4* __restrict__ a_bope,\n"
"    const float_sw4* __restrict__ a_ghcof, \n"
"    const float_sw4* __restrict__ a_acof_no_gp,\n"
"    const float_sw4* __restrict__ a_ghcof_no_gp, \n"
"    const float_sw4* __restrict__ a_strx,\n"
"    const float_sw4* __restrict__ a_stry ) \n"
"{\n"
"  int i = start0 + threadIdx.x + blockIdx.x * blockDim.x;\n"
"  int j = start1 + threadIdx.y + blockIdx.y * blockDim.y;\n"
"  int k = start2 + threadIdx.z + blockIdx.z * blockDim.z;\n"
"  if ((i < N0) && (j < N1) && (k < N2)) {\n"
"    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);\n"
"    float_sw4 istry = 1 / (stry(j));\n"
"    float_sw4 istrx = 1 / (strx(i));\n"
"    float_sw4 istrxy = istry * istrx;\n"
"\n"
"    float_sw4 r3 = 0.0;\n"
"\n"
"    // w-equation\n"
"\n"
"    //      r1 = 0;\n"
"    // pp derivative (w)\n"
"    // 43 ops, tot=1580\n"
"    float_sw4 cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *\n"
"      met(1, i - 2, j, k) * strx(i - 2);\n"
"    float_sw4 cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *\n"
"      met(1, i - 1, j, k) * strx(i - 1);\n"
"    float_sw4 cof3 =\n"
"      (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);\n"
"    float_sw4 cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *\n"
"      met(1, i + 1, j, k) * strx(i + 1);\n"
"    float_sw4 cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *\n"
"      met(1, i + 2, j, k) * strx(i + 2);\n"
"\n"
"    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);\n"
"    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);\n"
"    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);\n"
"    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);\n"
"\n"
"    r3 += i6 *\n"
"      (mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +\n"
"       mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +\n"
"       mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +\n"
"       mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *\n"
"      istry;\n"
"\n"
"    // qq derivative (w)\n"
"    // 43 ops, tot=1623\n"
"    {\n"
"      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;\n"
"      cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *\n"
"        met(1, i, j - 2, k) * stry(j - 2);\n"
"      cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *\n"
"        met(1, i, j - 1, k) * stry(j - 1);\n"
"      cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);\n"
"      cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *\n"
"        met(1, i, j + 1, k) * stry(j + 1);\n"
"      cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *\n"
"        met(1, i, j + 2, k) * stry(j + 2);\n"
"      mux1 = cof2 - tf * (cof3 + cof1);\n"
"      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);\n"
"      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);\n"
"      mux4 = cof4 - tf * (cof3 + cof5);\n"
"\n"
"      r3 += i6 *\n"
"        (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +\n"
"         mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +\n"
"         mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +\n"
"         mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *\n"
"        istrx;\n"
"    }\n"
"    // rr derivative (u)\n"
"    // 43 ops, tot=1666\n"
"    {\n"
"      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;\n"
"      cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *\n"
"        met(4, i, j, k - 2);\n"
"      cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *\n"
"        met(4, i, j, k - 1);\n"
"      cof3 =\n"
"        (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);\n"
"      cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *\n"
"        met(4, i, j, k + 1);\n"
"      cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *\n"
"        met(4, i, j, k + 2);\n"
"\n"
"      mux1 = cof2 - tf * (cof3 + cof1);\n"
"      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);\n"
"      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);\n"
"      mux4 = cof4 - tf * (cof3 + cof5);\n"
"\n"
"      r3 += i6 *\n"
"        (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +\n"
"         mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +\n"
"         mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +\n"
"         mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *\n"
"        istry;\n"
"    }\n"
"    // rr derivative (v)\n"
"    // 43 ops, tot=1709\n"
"    {\n"
"      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;\n"
"      cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *\n"
"        met(4, i, j, k - 2);\n"
"      cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *\n"
"        met(4, i, j, k - 1);\n"
"      cof3 =\n"
"        (mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * met(4, i, j, k);\n"
"      cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *\n"
"        met(4, i, j, k + 1);\n"
"      cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *\n"
"        met(4, i, j, k + 2);\n"
"\n"
"      mux1 = cof2 - tf * (cof3 + cof1);\n"
"      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);\n"
"      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);\n"
"      mux4 = cof4 - tf * (cof3 + cof5);\n"
"\n"
"      r3 += i6 *\n"
"        (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +\n"
"         mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +\n"
"         mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +\n"
"         mux4 * (u(2, i, j, k + 2) - u(2, i, j, k))) *\n"
"        istrx;\n"
"    }\n"
"\n"
"    // rr derivative (w)\n"
"    // 83 ops, tot=1792\n"
"    {\n"
"      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;\n"
"      cof1 = (2 * mu(i, j, k - 2) + la(i, j, k - 2)) *\n"
"        met(4, i, j, k - 2) * met(4, i, j, k - 2) +\n"
"        mu(i, j, k - 2) * (met(2, i, j, k - 2) * strx(i) *\n"
"            met(2, i, j, k - 2) * strx(i) +\n"
"            met(3, i, j, k - 2) * stry(j) *\n"
"            met(3, i, j, k - 2) * stry(j));\n"
"      cof2 = (2 * mu(i, j, k - 1) + la(i, j, k - 1)) *\n"
"        met(4, i, j, k - 1) * met(4, i, j, k - 1) +\n"
"        mu(i, j, k - 1) * (met(2, i, j, k - 1) * strx(i) *\n"
"            met(2, i, j, k - 1) * strx(i) +\n"
"            met(3, i, j, k - 1) * stry(j) *\n"
"            met(3, i, j, k - 1) * stry(j));\n"
"      cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(4, i, j, k) *\n"
"        met(4, i, j, k) +\n"
"        mu(i, j, k) *\n"
"        (met(2, i, j, k) * strx(i) * met(2, i, j, k) * strx(i) +\n"
"         met(3, i, j, k) * stry(j) * met(3, i, j, k) * stry(j));\n"
"      cof4 = (2 * mu(i, j, k + 1) + la(i, j, k + 1)) *\n"
"        met(4, i, j, k + 1) * met(4, i, j, k + 1) +\n"
"        mu(i, j, k + 1) * (met(2, i, j, k + 1) * strx(i) *\n"
"            met(2, i, j, k + 1) * strx(i) +\n"
"            met(3, i, j, k + 1) * stry(j) *\n"
"            met(3, i, j, k + 1) * stry(j));\n"
"      cof5 = (2 * mu(i, j, k + 2) + la(i, j, k + 2)) *\n"
"        met(4, i, j, k + 2) * met(4, i, j, k + 2) +\n"
"        mu(i, j, k + 2) * (met(2, i, j, k + 2) * strx(i) *\n"
"            met(2, i, j, k + 2) * strx(i) +\n"
"            met(3, i, j, k + 2) * stry(j) *\n"
"            met(3, i, j, k + 2) * stry(j));\n"
"      mux1 = cof2 - tf * (cof3 + cof1);\n"
"      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);\n"
"      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);\n"
"      mux4 = cof4 - tf * (cof3 + cof5);\n"
"\n"
"      r3 +=\n"
"        i6 *\n"
"        (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +\n"
"         mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +\n"
"         mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +\n"
"         mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *\n"
"        istrxy\n"
"        // pr-derivatives\n"
"        // 86 ops, tot=1878\n"
"        // r1 +=\n"
"        +\n"
"        c2 *\n"
"        ((la(i, j, k + 2)) * met(4, i, j, k + 2) *\n"
"         met(1, i, j, k + 2) *\n"
"         (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +\n"
"          c1 *\n"
"          (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *\n"
"         istry +\n"
"         mu(i, j, k + 2) * met(2, i, j, k + 2) *\n"
"         met(1, i, j, k + 2) *\n"
"         (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +\n"
"          c1 *\n"
"          (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *\n"
"         strx(i) * istry -\n"
"         ((la(i, j, k - 2)) * met(4, i, j, k - 2) *\n"
"          met(1, i, j, k - 2) *\n"
"          (c2 *\n"
"           (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +\n"
"           c1 * (u(1, i + 1, j, k - 2) -\n"
"             u(1, i - 1, j, k - 2))) *\n"
"          istry +\n"
"          mu(i, j, k - 2) * met(2, i, j, k - 2) *\n"
"          met(1, i, j, k - 2) *\n"
"          (c2 *\n"
"           (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +\n"
"           c1 * (u(3, i + 1, j, k - 2) -\n"
"             u(3, i - 1, j, k - 2))) *\n"
"          strx(i) * istry)) +\n"
"          c1 *\n"
"          ((la(i, j, k + 1)) * met(4, i, j, k + 1) *\n"
"           met(1, i, j, k + 1) *\n"
"           (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +\n"
"            c1 *\n"
"            (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *\n"
"           istry +\n"
"           mu(i, j, k + 1) * met(2, i, j, k + 1) *\n"
"           met(1, i, j, k + 1) *\n"
"           (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +\n"
"            c1 *\n"
"            (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *\n"
"           strx(i) * istry -\n"
"           (la(i, j, k - 1) * met(4, i, j, k - 1) *\n"
"            met(1, i, j, k - 1) *\n"
"            (c2 *\n"
"             (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +\n"
"             c1 * (u(1, i + 1, j, k - 1) -\n"
"               u(1, i - 1, j, k - 1))) *\n"
"            istry +\n"
"            mu(i, j, k - 1) * met(2, i, j, k - 1) *\n"
"            met(1, i, j, k - 1) *\n"
"            (c2 *\n"
"             (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +\n"
"             c1 * (u(3, i + 1, j, k - 1) -\n"
"               u(3, i - 1, j, k - 1))) *\n"
"            strx(i) * istry))\n"
"            // rp derivatives\n"
"            // 79 ops, tot=1957\n"
"            //   r1 +=\n"
"            + istry * (c2 * ((mu(i + 2, j, k)) * met(4, i + 2, j, k) *\n"
"                  met(1, i + 2, j, k) *\n"
"                  (c2 * (u(1, i + 2, j, k + 2) -\n"
"                   u(1, i + 2, j, k - 2)) +\n"
"                   c1 * (u(1, i + 2, j, k + 1) -\n"
"                     u(1, i + 2, j, k - 1))) +\n"
"                  mu(i + 2, j, k) * met(2, i + 2, j, k) *\n"
"                  met(1, i + 2, j, k) *\n"
"                  (c2 * (u(3, i + 2, j, k + 2) -\n"
"                   u(3, i + 2, j, k - 2)) +\n"
"                   c1 * (u(3, i + 2, j, k + 1) -\n"
"                     u(3, i + 2, j, k - 1))) *\n"
"                  strx(i + 2) -\n"
"                  (mu(i - 2, j, k) * met(4, i - 2, j, k) *\n"
"                   met(1, i - 2, j, k) *\n"
"                   (c2 * (u(1, i - 2, j, k + 2) -\n"
"                    u(1, i - 2, j, k - 2)) +\n"
"                    c1 * (u(1, i - 2, j, k + 1) -\n"
"                      u(1, i - 2, j, k - 1))) +\n"
"                   mu(i - 2, j, k) * met(2, i - 2, j, k) *\n"
"                   met(1, i - 2, j, k) *\n"
"                   (c2 * (u(3, i - 2, j, k + 2) -\n"
"                    u(3, i - 2, j, k - 2)) +\n"
"                    c1 * (u(3, i - 2, j, k + 1) -\n"
"                      u(3, i - 2, j, k - 1))) *\n"
"                   strx(i - 2))) +\n"
"                   c1 * ((mu(i + 1, j, k)) * met(4, i + 1, j, k) *\n"
"                       met(1, i + 1, j, k) *\n"
"                       (c2 * (u(1, i + 1, j, k + 2) -\n"
"                        u(1, i + 1, j, k - 2)) +\n"
"                        c1 * (u(1, i + 1, j, k + 1) -\n"
"                          u(1, i + 1, j, k - 1))) +\n"
"                       mu(i + 1, j, k) * met(2, i + 1, j, k) *\n"
"                       met(1, i + 1, j, k) *\n"
"                       (c2 * (u(3, i + 1, j, k + 2) -\n"
"                        u(3, i + 1, j, k - 2)) +\n"
"                        c1 * (u(3, i + 1, j, k + 1) -\n"
"                          u(3, i + 1, j, k - 1))) *\n"
"                       strx(i + 1) -\n"
"                       (mu(i - 1, j, k) * met(4, i - 1, j, k) *\n"
"                        met(1, i - 1, j, k) *\n"
"                        (c2 * (u(1, i - 1, j, k + 2) -\n"
"                         u(1, i - 1, j, k - 2)) +\n"
"                         c1 * (u(1, i - 1, j, k + 1) -\n"
"                           u(1, i - 1, j, k - 1))) +\n"
"                        mu(i - 1, j, k) * met(2, i - 1, j, k) *\n"
"                        met(1, i - 1, j, k) *\n"
"                        (c2 * (u(3, i - 1, j, k + 2) -\n"
"                         u(3, i - 1, j, k - 2)) +\n"
"                         c1 * (u(3, i - 1, j, k + 1) -\n"
"                           u(3, i - 1, j, k - 1))) *\n"
"                        strx(i - 1))))\n"
"                        // qr derivatives\n"
"                        // 86 ops, tot=2043\n"
"                        //     r1 +=\n"
"                        +\n"
"                        c2 *\n"
"                        (mu(i, j, k + 2) * met(3, i, j, k + 2) *\n"
"                         met(1, i, j, k + 2) *\n"
"                         (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j - 2, k + 2)) +\n"
"                    c1 *\n"
"                    (u(3, i, j + 1, k + 2) - u(3, i, j - 1, k + 2))) *\n"
"                         stry(j) * istrx +\n"
"                         la(i, j, k + 2) * met(4, i, j, k + 2) *\n"
"                         met(1, i, j, k + 2) *\n"
"                         (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +\n"
"                    c1 *\n"
"                    (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) *\n"
"                         istrx -\n"
"                         (mu(i, j, k - 2) * met(3, i, j, k - 2) *\n"
"                    met(1, i, j, k - 2) *\n"
"                    (c2 *\n"
"                     (u(3, i, j + 2, k - 2) - u(3, i, j - 2, k - 2)) +\n"
"                     c1 * (u(3, i, j + 1, k - 2) -\n"
"                       u(3, i, j - 1, k - 2))) *\n"
"                    stry(j) * istrx +\n"
"                    la(i, j, k - 2) * met(4, i, j, k - 2) *\n"
"                    met(1, i, j, k - 2) *\n"
"                    (c2 *\n"
"                     (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +\n"
"                     c1 * (u(2, i, j + 1, k - 2) -\n"
"                       u(2, i, j - 1, k - 2))) *\n"
"                    istrx)) +\n"
"                    c1 *\n"
"                    (mu(i, j, k + 1) * met(3, i, j, k + 1) *\n"
"                     met(1, i, j, k + 1) *\n"
"                     (c2 * (u(3, i, j + 2, k + 1) - u(3, i, j - 2, k + 1)) +\n"
"                      c1 *\n"
"                      (u(3, i, j + 1, k + 1) - u(3, i, j - 1, k + 1))) *\n"
"                     stry(j) * istrx +\n"
"                     la(i, j, k + 1) * met(4, i, j, k + 1) *\n"
"                     met(1, i, j, k + 1) *\n"
"                     (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +\n"
"                      c1 *\n"
"                      (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) *\n"
"                     istrx -\n"
"                     (mu(i, j, k - 1) * met(3, i, j, k - 1) *\n"
"                      met(1, i, j, k - 1) *\n"
"                      (c2 *\n"
"                       (u(3, i, j + 2, k - 1) - u(3, i, j - 2, k - 1)) +\n"
"                       c1 * (u(3, i, j + 1, k - 1) -\n"
"                         u(3, i, j - 1, k - 1))) *\n"
"                      stry(j) * istrx +\n"
"                      la(i, j, k - 1) * met(4, i, j, k - 1) *\n"
"                      met(1, i, j, k - 1) *\n"
"                      (c2 *\n"
"                       (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +\n"
"                       c1 * (u(2, i, j + 1, k - 1) -\n"
"                         u(2, i, j - 1, k - 1))) *\n"
"                      istrx))\n"
"                      // rq derivatives\n"
"                      //  79 ops, tot=2122\n"
"                      //  r1 +=\n"
"                      + istrx * (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *\n"
"                            met(1, i, j + 2, k) *\n"
"                            (c2 * (u(3, i, j + 2, k + 2) -\n"
"                             u(3, i, j + 2, k - 2)) +\n"
"                             c1 * (u(3, i, j + 2, k + 1) -\n"
"                               u(3, i, j + 2, k - 1))) *\n"
"                            stry(j + 2) +\n"
"                            mu(i, j + 2, k) * met(4, i, j + 2, k) *\n"
"                            met(1, i, j + 2, k) *\n"
"                            (c2 * (u(2, i, j + 2, k + 2) -\n"
"                             u(2, i, j + 2, k - 2)) +\n"
"                             c1 * (u(2, i, j + 2, k + 1) -\n"
"                               u(2, i, j + 2, k - 1))) -\n"
"                            (mu(i, j - 2, k) * met(3, i, j - 2, k) *\n"
"                             met(1, i, j - 2, k) *\n"
"                             (c2 * (u(3, i, j - 2, k + 2) -\n"
"                              u(3, i, j - 2, k - 2)) +\n"
"                              c1 * (u(3, i, j - 2, k + 1) -\n"
"                                u(3, i, j - 2, k - 1))) *\n"
"                             stry(j - 2) +\n"
"                             mu(i, j - 2, k) * met(4, i, j - 2, k) *\n"
"                             met(1, i, j - 2, k) *\n"
"                             (c2 * (u(2, i, j - 2, k + 2) -\n"
"                              u(2, i, j - 2, k - 2)) +\n"
"                              c1 * (u(2, i, j - 2, k + 1) -\n"
"                                u(2, i, j - 2, k - 1))))) +\n"
"                                c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *\n"
"                                    met(1, i, j + 1, k) *\n"
"                                    (c2 * (u(3, i, j + 1, k + 2) -\n"
"                                     u(3, i, j + 1, k - 2)) +\n"
"                                     c1 * (u(3, i, j + 1, k + 1) -\n"
"                                       u(3, i, j + 1, k - 1))) *\n"
"                                    stry(j + 1) +\n"
"                                    mu(i, j + 1, k) * met(4, i, j + 1, k) *\n"
"                                    met(1, i, j + 1, k) *\n"
"                                    (c2 * (u(2, i, j + 1, k + 2) -\n"
"                                     u(2, i, j + 1, k - 2)) +\n"
"                                     c1 * (u(2, i, j + 1, k + 1) -\n"
"                                       u(2, i, j + 1, k - 1))) -\n"
"                                    (mu(i, j - 1, k) * met(3, i, j - 1, k) *\n"
"                                     met(1, i, j - 1, k) *\n"
"                                     (c2 * (u(3, i, j - 1, k + 2) -\n"
"                                      u(3, i, j - 1, k - 2)) +\n"
"                                      c1 * (u(3, i, j - 1, k + 1) -\n"
"                                        u(3, i, j - 1, k - 1))) *\n"
"                                     stry(j - 1) +\n"
"                                     mu(i, j - 1, k) * met(4, i, j - 1, k) *\n"
"                                     met(1, i, j - 1, k) *\n"
"                                     (c2 * (u(2, i, j - 1, k + 2) -\n"
"                                      u(2, i, j - 1, k - 2)) +\n"
"                                      c1 * (u(2, i, j - 1, k + 1) -\n"
"                                        u(2, i, j - 1, k - 1))))));\n"
"    }\n"
"\n"
"    // 4 ops, tot=2126\n"
"    lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;\n"
"  }\n"
"}\n"
"\n"
;
