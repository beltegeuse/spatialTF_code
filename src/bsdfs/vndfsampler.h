#pragma once

#include <mitsuba/mitsuba.h>
#include <math.h>

MTS_NAMESPACE_BEGIN

/*----------------------------------------------------------------------------*/
#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

#ifdef WIN32
inline double erfinv (double x)
{
  SLog(EError, "undefined function erf()\n");
  return (NAN);
}
#else
inline double erfinv(double x) {
  double x2, r, y;
  int sign_x;

  if (x < -1 || x > 1)
    return NAN;

  if (x == 0)
    return 0;

  if (x > 0)
    sign_x = 1;
  else {
    sign_x = -1;
    x = -x;
  }

  if (x <= 0.7) {

    x2 = x * x;
    r = x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
    r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 +
    erfinv_b1) * x2 + erfinv_b0;
  } else {
    y = sqrt(-log((1 - x) / 2));
    r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
    r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
  }

  r = r * sign_x;
  x = x * sign_x;

  r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));
  r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));

  return r;
}
#endif

#undef erfinv_a3
#undef erfinv_a2
#undef erfinv_a1
#undef erfinv_a0

#undef erfinv_b4
#undef erfinv_b3
#undef erfinv_b2
#undef erfinv_b1
#undef erfinv_b0

#undef erfinv_c3
#undef erfinv_c2
#undef erfinv_c1
#undef erfinv_c0

#undef erfinv_d2
#undef erfinv_d1
#undef erfinv_d0

//////////////////////
// General sampler
// to sample the normal visible distribution
// this is based on Precomputed tables
//////////////////////
class VNDFSampler : public Object {
 public:
  VNDFSampler();

  // generate a view dependent normal
  void sample(
      // input
      const double omega_i[3],  // incident direction
      const double alpha_x, const double alpha_y,  // anisotropic roughness
      const double U1, const double U2,  // random numbers
      // output
      double omega_m[3]  // normal
      ) const;

  MTS_DECLARE_CLASS()
 protected:

  virtual ~VNDFSampler();

  // method implemented by child class
  // inverse slope CDF
  virtual void sample11(
      // input
      const double theta_i,  // incident direction
      double U1, double U2,  // random numbers
      // output
      double& slope_x, double& slope_y  // slopes
      ) const = 0;
};

//////////////////////
// The implementation
// of the beckman sampler
//
//////////////////////

class VNDFSamplerBeckmann : public VNDFSampler {
 protected:

  ////////////////////////
  // Methods used to compute the distribution
  ////////////////////////

  virtual void sample11(
  // input
      const double theta_i,  // incident direction
      double U1, double U2,  // random numbers
      // output
      double& slope_x, double& slope_y  // slopes
      ) const {

    // special case (normal incidence)
    if (theta_i < 0.0001) {
      const double r = sqrt(-log(U1));
      const double phi = 6.28318530718 * U2;
      slope_x = r * cos(phi);
      slope_y = r * sin(phi);
      return;
    }

    // precomputations
    const double sin_theta_i = sin(theta_i);
    const double cos_theta_i = cos(theta_i);
    const double tan_theta_i = sin_theta_i / cos_theta_i;
    const double a = 1.0 / tan_theta_i;
    const double erf_a = erf(a);
    const double exp_a2 = exp(-a * a);
    const double SQRT_PI_INV = 0.56418958354;
    const double Lambda = 0.5 * (erf_a - 1) + 0.5 * SQRT_PI_INV * exp_a2 / a;
    const double G1 = 1.0 / (1.0 + Lambda);  // masking
    const double C = 1.0 - G1 * erf_a;
    // sample slope X
    if (U1 < C) {
      // rescale U1
      U1 = U1 / C;
      const double w_1 = 0.5 * SQRT_PI_INV * sin_theta_i * exp_a2;
      const double w_2 = cos_theta_i * (0.5 - 0.5 * erf_a);
      const double p = w_1 / (w_1 + w_2);
      if (U1 < p) {
        U1 = U1 / p;
        slope_x = -sqrt(-log(U1 * exp_a2));
      } else {
        U1 = (U1 - p) / (1.0 - p);
        slope_x = erfinv(U1 - 1.0 - U1 * erf_a);
      }
    } else {
      // rescale U1
      U1 = (U1 - C) / (1.0 - C);
      slope_x = erfinv((-1.0 + 2.0 * U1) * erf_a);
      const double p = (-slope_x * sin_theta_i + cos_theta_i)
          / (2.0 * cos_theta_i);
      if (U2 > p) {
        slope_x = -slope_x;
        U2 = (U2 - p) / (1.0 - p);
      } else
        U2 = U2 / p;
    }
    // sample slope Y
    slope_y = erfinv(2.0 * U2 - 1.0);
  }
};

MTS_NAMESPACE_END

