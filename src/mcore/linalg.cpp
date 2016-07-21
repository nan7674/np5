# include "linalg.hpp"

# include <algorithm>
# include <cmath>
# include <cstring>
# include <limits>

# include "calc.hpp"

namespace {

	using ::mcore::calc::sqr;

	class allocator {
	public:
		static allocator& instance() noexcept {
			static allocator obj;
			return obj;
		}

		void* allocate(size_t nbytes) {
			if (nbytes > nbytes_) {
				buffer_.reset(new unsigned char[nbytes]);
				nbytes_ = nbytes;
			}
			return buffer_.get();
		}

		void release(void* ptr) {
		}

	private:
		size_t nbytes_{0};
		std::unique_ptr<unsigned char[]> buffer_;
	};

} // anonymous namespace


mcore::linalg::mat mcore::linalg::mat::copy() const {
	return mat(rows_, cols_, data_.copy());
}


bool mcore::linalg::eq(vec const& x, vec const& y, double tol) noexcept {
	if (x.dim() == y.dim()) {
		for (size_t i = 0; i < x.dim(); ++i)
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



/*
// Solve Ax = b
// Currently the uses the most stupid Gauss method
mcore::linalg::vec mcore::linalg::solve(mat const& A, vec const& y) {
	assert(A.rows() == A.cols());
	assert(A.rows() == y.dim());

	size_t const dim = A.rows();

	// Create permutation
	double* perm = static_cast<double*>(
		allocator::instance().allocate(sizeof(double) * dim));
	//std::unique_ptr<double[]> perm{new double[dim]};
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

	::mcore::detail::sequence<double> r1(dim);
	for (size_t i = 0; i < dim; ++i)
		r1[i] = r(perm[i]);

	return vec(std::move(r1));
}
*/


inline size_t mcore::linalg::leq_solver::get_memory_size(size_t dim) noexcept {
	return sizeof(double) * (dim * dim + dim) + sizeof(size_t) * dim;
}

mcore::linalg::leq_solver::leq_solver(size_t dim)
	: dim_(dim),
	  A_(dim), rhs_(dim),
	  permutation_(dim),
	  data_(new uint8_t[get_memory_size(dim)])
{
	A_.data_ = reinterpret_cast<double*>(data_.get());
	rhs_.data_ = A_.data_ + dim_ * dim_;
	permutation_.data_ = reinterpret_cast<size_t*>(rhs_.data_ + dim_);
}

inline void mcore::linalg::leq_solver::init_permutation() noexcept {
	for (size_t i = 0; i < dim_; ++i)
		permutation_[i] = i;
}

void
mcore::linalg::leq_solver::triangulate(mat const& A, vec const& rhs) noexcept {
	A_.copy_from(A.data().data());
	rhs_.copy_from(rhs.data().data());
	init_permutation();

	for (size_t i = 0; i < dim_; ++i) {
		size_t max_index = i;
		double max_value = std::abs(A_(permutation_[i], i));
		for (size_t j = i + 1; j < dim_; ++j) {
			if (std::abs(A_(permutation_[j], i)) > max_value) {
				max_value = std::abs(A_(permutation_[j], i));
				max_index = j;
			}
		}
		std::swap(permutation_[i], permutation_[max_index]);
		max_value = A_(permutation_[i], i);

		for (size_t j = i; j < dim_; ++j)
			A_(permutation_[i], j) /= max_value;
		rhs_(permutation_[i]) /= max_value;

		for (size_t j = i + 1; j < dim_; ++j) {
			double const K = A_(permutation_[j], i);
			for (size_t k = i; k < dim_; ++k)
				A_(permutation_[j], k) -= K * A_(permutation_[i], k);
			rhs_(permutation_[j]) -= K * rhs_(permutation_[i]);
		}
	}

	for (size_t i = dim_ - 2; i < dim_; --i) {
		double s = 0;
		for (size_t j = i + 1; j < dim_; ++j)
			s += A_(permutation_[i], j) * rhs_(permutation_[j]);
		rhs_(permutation_[i]) -= s;
	}
}


mcore::linalg::vec
mcore::linalg::leq_solver::solve(mat const& A, vec const& rhs) {
	triangulate(A, rhs);
	mcore::linalg::vec ret(dim_);
	for (size_t i = 0; i < dim_; ++i)
		ret[i] = rhs_(permutation_[i]);
	return ret;
}

void mcore::linalg::leq_solver::solve(mat const& A, vec const& rhs, vec& sol) {
	assert(rhs.dim() == sol.dim());
	assert(rhs.dim() == dim_);
	triangulate(A, rhs);
	for (size_t i = 0; i < dim_; ++i)
		sol[i] = rhs_(permutation_[i]);
}


// =============================================================================
// Low-level matrix oprations
void mcore::linalg::solve_tridiagonal(
	double* const __restrict__ d0,
	double* const __restrict__ d1,
	double* const __restrict__ d2,
	double* const __restrict__ x,
	const size_t dim) noexcept
{
	d2[0] /= d1[0];
	x[0] /= d1[0];

	for (size_t i = 1; i < dim; i++) {
		double const m = 1.0f / (d1[i] - d0[i] * d2[i - 1]);
		d2[i] *= m;
		x[i] = (x[i] - d0[i] * x[i - 1]) * m;
	}

	for (size_t i = dim - 1; i-- > 0; )
		x[i] += -d2[i] * x[i + 1];
}
