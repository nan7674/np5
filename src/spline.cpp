# include "spline.hpp"


void np5::spline_boundary::fill_fringe(
	double* const __restrict__ /*diag*/,
	double* /*bs*/,
	spline_fringe const& /* boundary */,
	size_t /* size */) const noexcept {}


/*! \brief Update spline coefficients
 */
void np5::spline_boundary::update_coefficients(
	double* const __restrict__ cs,
	spline_fringe const& /* boundary */,
	size_t size) const noexcept
{
	cs[0] = 0;
	cs[size - 1] = 0;
}


/*! @brief Fills boundary condition for splines
 */
void np5::d2_boundary::fill_fringe(
	double* const __restrict__ diag,
	double* const __restrict__ bs,
	spline_fringe const& __restrict__ boundary,
	size_t const size) const noexcept
{
	double const q0 = boundary.h0 * m0_;
	double const qn = boundary.hn * m1_;

	bs[1] -= q0;
	bs[size - 2] -= qn;
}


/*! \brief Update spline coefficients
 */
void np5::d2_boundary::update_coefficients(
	double* __restrict__ cs,
	spline_fringe const& __restrict__ boundary,
	size_t size) const noexcept
{
	cs[0] = m0_;
	cs[size - 1] = m1_;
}


inline double np5::d1_boundary::get_k0(spline_fringe const& bc) const noexcept {
	return 1.5 * ((bc.f1 - bc.f0) / bc.h0 - fp0_);
}


inline double np5::d1_boundary::get_kn(spline_fringe const& bc) const noexcept {
	return 1.5 * (fp1_ - (bc.g1 - bc.g0) / bc.hn);
}


/*! @brief Fills boundary condition for splines
 */
void np5::d1_boundary::fill_fringe(
	double* const __restrict__ diag,
	double* const __restrict__ bs,
	spline_fringe const& __restrict__ bc,
	size_t const size) const noexcept
{
	double const K0 = get_k0(bc);
	double const q0 = 0.5 * bc.h0;

	diag[0] -= q0;
	bs[1] -= K0;

	double const Kn = get_kn(bc);
	double const qn = 0.5 * bc.hn;

	diag[size - 3] -= qn;
	bs[size - 2] -= Kn;
}


/*! \brief Update spline coefficients
 */
void np5::d1_boundary::update_coefficients(
	double* cs,
	spline_fringe const& bc,
	size_t size) const noexcept
{
	cs[0] = (get_k0(bc) - 0.5 * cs[1] * bc.h0) / bc.h0;
	cs[size - 1] = (get_kn(bc) - 0.5 * cs[size - 2] * bc.hn) / bc.hn;
}
