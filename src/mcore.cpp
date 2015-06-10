# include "mcore.hpp"


namespace mcore = np5::mcore;

/* @brief Cholesky decomposition of a symmetric matrices
 *
 * @param d0 a main diagonal of the matrix
 */
void mcore::cholesky(
		double* const d0,
		double* const d1,
		double* const d2,
		size_t const n) noexcept {

	d1[0] /= d0[0];
	d2[0] /= d0[0];

	d0[1] -= sqr(d1[0]) * d0[0];
	d1[1] = (d1[1] - d0[0] * d1[0] * d2[0]) / d0[1];
	d2[1] /= d0[1];

	for (size_t i = 2; i < n - 2; ++i) {
		d0[i] -= d0[i - 2] * sqr(d2[i - 2]) + d0[i - 1] * sqr(d1[i - 1]);
		d1[i] = (d1[i] - d0[i - 1] * d1[i - 1] * d2[i - 1]) / d0[i];
		d2[i] /= d0[i];
	}

	d0[n - 2] -= d0[n - 4] * sqr(d2[n - 4]) + d0[n - 3] * sqr(d1[n - 3]);
	d1[n - 2] = (d1[n - 2] - d0[n - 3] * d1[n - 3] * d2[n - 3]) / d0[n - 2];

	d0[n - 1] -= d0[n - 3] * sqr(d2[n - 3]) + d0[n - 2] * sqr(d1[n - 2]);
}


/* @brief Cholesky decomposition of a symmetric matrices
*
* @param d0 a main diagonal of the matrix
*/
void mcore::cholesky(
		double* const d0,
		double* const d1,
		size_t const n) noexcept{

	d1[0] /= d0[0];

	d0[1] -= sqr(d1[0]) * d0[0];
	d1[1] /= d0[1];

	for (size_t i = 2; i < n - 2; ++i) {
		d0[i] -= d0[i - 1] * sqr(d1[i - 1]);
		d1[i] /= d0[i];
	}

	d0[n - 2] -= d0[n - 3] * sqr(d1[n - 3]);
	d1[n - 2] /= d0[n - 2];

	d0[n - 1] -= d0[n - 2] * sqr(d1[n - 2]);
}


void mcore::solve_ldl(
		double const* const d0,
		double const* const d1,
		double const* const d2,
		double* const y,
		size_t const n) noexcept {
	y[1] -= d1[0] * y[0];

	for (size_t i = 2; i < n; ++i)
		y[i] -= d1[i - 1] * y[i - 1] + d2[i - 2] * y[i - 2];

	for (size_t i = 0; i < n; ++i)
		y[i] /= d0[i];

	y[n - 2] -= d1[n - 2] * y[n - 1];

	for (size_t i = n - 3; i < std::numeric_limits<size_t>::max(); --i)
		y[i] -= d1[i] * y[i + 1] + d2[i] * y[i + 2];
}


void mcore::solve_ldl(
		double const* const d0,
		double const* const d1,
		double* const y,
		size_t const n) noexcept{
	y[1] -= d1[0] * y[0];

	for (size_t i = 2; i < n; ++i)
		y[i] -= d1[i - 1] * y[i - 1];

	for (size_t i = 0; i < n; ++i)
		y[i] /= d0[i];

	y[n - 2] -= d1[n - 2] * y[n - 1];

	for (size_t i = n - 3; i < std::numeric_limits<size_t>::max(); --i)
		y[i] -= d1[i] * y[i + 1];
}
