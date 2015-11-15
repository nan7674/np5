# include "linalg.hpp"

# include <algorithm>
# include <cmath>
# include <cstring>
# include <limits>

namespace {

	template <typename T>
	T sqr(T value) noexcept { return value * value; }

} // anonymous namespace


mcore::linalg::vec mcore::linalg::vec::copy() const {
	vec other;
	other.data_.reset(new double[dim_]);
	other.dim_ = dim_;

	std::memcpy(other.data_.get(), data_.get(), sizeof(value_type) * dim_);

	return other;
}

void mcore::linalg::vec::clear() {
	if (dim_)
		std:memset(data_.get(), 0, sizeof(double) * dim_);
}

mcore::linalg::mat mcore::linalg::mat::copy() const {
	mat other;

	other.data_.reset(new double[row_ * col_]);
	other.row_ = row_;
	other.col_ = col_;

	std::memcpy(other.data_.get(), data_.get(), sizeof(value_type) * row_ * col_);

	return other;
}

bool mcore::linalg::eq(vec const& x, vec const& y, double tol) noexcept {
	if (x.rows() == y.rows()) {
		for (size_t i = 0; i < x.rows(); ++i)
			if (std::abs(x(i) - y(i)) > tol)
				return false;
		return true;
	} else
		return false;
}

bool mcore::linalg::eq(mat const& x, mat const& y, double tol) noexcept {
	if (x.rows() == y.rows() && x.cols() == y.cols()) {
		for (size_t i = 0; i < x.rows(); ++i)
			for (size_t j = 0; j < x.cols(); ++j)
				if (std::abs(x(i, j) - y(i, j)) > tol)
					return false;
		return true;
	} else
		return false;
}




// Solve Ax = b
// Currently the uses the most stupid Gauss method
mcore::linalg::vec mcore::linalg::solve(mat const& A, vec const& y) {
	assert(A.rows() == A.cols());
	assert(A.rows() == y.rows());

	size_t const dim = A.rows();

	// Create permutation
	std::unique_ptr<double[]> perm{new double[dim]};
	for (size_t i = 0; i < dim; ++i)
		perm[i] = i;

	// Create wrking copy of the data
	vec r = y.copy();
	mat a = A.copy();

	for (size_t i = 0; i < dim; ++i) {
		size_t max_index = i;
		double max_value = std::abs(a(perm[i], i));
		for (size_t j = i + 1; j < dim; ++j) {
			if (std::abs(a(perm[j], i)) > max_value) {
				max_value = std::abs(a(perm[j], i));
				max_index = j;
			}
		}
		std::swap(perm[i], perm[max_index]);
		max_value = a(perm[i], i);

		for (size_t j = i; j < dim; ++j)
			a(perm[i], j) /= max_value;
		r(perm[i]) /= max_value;

		for (size_t j = i + 1; j < dim; ++j) {
			double const K = a(perm[j], i);
			for (size_t k = i; k < dim; ++k)
				a(perm[j], k) -= K * a(perm[i], k);
			r(perm[j]) -= K * r(perm[i]);
		}
	}

	for (size_t i = dim - 2; i < std::numeric_limits<size_t>::max(); --i) {
		double s = 0;
		for (size_t j = i + 1; j < dim; ++j)
			s += a(perm[i], j) * r(perm[j]);
		r(perm[i]) -= s;
	}

	vec r1(dim);
	for (size_t i = 0; i < dim; ++i)
		r1(i) = r(perm[i]);

	return r1;
}

// =============================================================================
// Low-level matrix oprations

/* @brief Cholesky decomposition of a symmetric matrices
 *
 * @param d0 a main diagonal of the matrix
 */
void mcore::linalg::cholesky(
		double* const __restrict__ d0,
		double* const __restrict__ d1,
		double* const __restrict__ d2,
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
void mcore::linalg::cholesky(
		double* const __restrict__ d0,
		double* const __restrict__ d1,
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


void mcore::linalg::solve_ldl(
		double const* const __restrict__ d0,
		double const* const __restrict__ d1,
		double const* const __restrict__ d2,
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


void mcore::linalg::solve_ldl(
		double const* const __restrict__ d0,
		double const* const __restrict__ d1,
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

