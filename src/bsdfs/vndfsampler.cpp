#include "vndfsampler.h"

MTS_NAMESPACE_BEGIN

//////////////////////////////////
//////////////////////////////////
// Sampler VNDF
//////////////////////////////////
//////////////////////////////////

MTS_IMPLEMENT_CLASS(VNDFSampler, false, Object);

VNDFSampler::VNDFSampler() {
//  SLog(EInfo, "Create new sampler !!!");
}
VNDFSampler::~VNDFSampler() {
//  SLog(EWarn, "Delete sampler !!!");
}
void VNDFSampler::sample(
// input
    const double omega_i[3],  // incident direction
    const double alpha_x, const double alpha_y,  // anisotropic roughness
    const double U1, const double U2,  // random numbers
// output
    double omega_m[3]) const // micronormal
    {
// 1. stretch omega_i
  double omega_i_[3];
  omega_i_[0] = alpha_x * omega_i[0];
  omega_i_[1] = alpha_y * omega_i[1];
  omega_i_[2] = omega_i[2];
// normalize
  double inv_omega_i = 1.0
      / sqrt(
          omega_i_[0] * omega_i_[0] + omega_i_[1] * omega_i_[1]
              + omega_i_[2] * omega_i_[2]);
  omega_i_[0] *= inv_omega_i;
  omega_i_[1] *= inv_omega_i;
  omega_i_[2] *= inv_omega_i;
// get polar coordinates of omega_i_
  double theta_ = 0.0;
  double phi_ = 0.0;
  if (omega_i_[2] < 0.99999) {
    theta_ = acos(omega_i_[2]);
    phi_ = atan2(omega_i_[1], omega_i_[0]);
  }
// 2. sample P22_{omega_i}(x_slope, y_slope, 1, 1)
  double slope_x, slope_y;
  sample11(theta_, U1, U2, slope_x, slope_y);
// 3. rotate
  double tmp = cos(phi_) * slope_x - sin(phi_) * slope_y;
  slope_y = sin(phi_) * slope_x + cos(phi_) * slope_y;
  slope_x = tmp;
// 4. unstretch
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;
// 5. compute normal
  double inv_omega_m = sqrt(slope_x * slope_x + slope_y * slope_y + 1.0);
  omega_m[0] = -slope_x / inv_omega_m;
  omega_m[1] = -slope_y / inv_omega_m;
  omega_m[2] = 1.0 / inv_omega_m;
}
MTS_NAMESPACE_END
