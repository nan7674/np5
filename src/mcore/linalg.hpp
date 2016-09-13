# pragma once

# include <cassert>
# include <memory>
# include <cstring>
# include <cstddef>
# include <initializer_list>

# include "config.hpp"
# include "sequence.hpp"

# ifdef WITH_LAPACK
# define GET_ELEMENT(data, nrows, ncols, row, column) data[(column) * (nrows) + (row)]
# else
# define GET_ELEMENT(data, nrows, ncols, row, column) data[(row) * (ncols) + (column)]
# endif

namespace mcore { namespace linalg {

	class mat;
	class vec;
	class leq_solver;

	class vec {
		vec(vec&) = delete;
		vec& operator=(vec&) = delete;

	public:
		typedef double value_type;
		friend vec solve(mat const&, vec const&);

	private:
		typedef ::mcore::detail::sequence<value_type> container_type;

	public:
		/** @brief Creates empty vector.
		 *
		 * The vector has 0 dimension.
		 */
		vec() noexcept {}

		/*! @brief Creates of a given dimension.
		 *
		 * @param dim dimension of a vector.
		 *
		 * The constructor creates a vector with a dimension dim.
		 * Memory for the data is resrved but doesn't clear.s
		 */
		vec(size_t dim) : data_(dim) {}

		/** @brief Creates vector of a given dimension
		 */
		vec(size_t dim, double elem)
			: data_(dim, elem) {}

		vec(std::initializer_list<double> L)
			: data_(L) {}

		vec(vec&& other) noexcept
			: data_(std::move(other.data_)) {}

		vec& operator=(vec&& other) noexcept {
			if (this != &other) {
				data_ = std::move(other.data_);
			}
			return *this;
		}

		template <typename E>
		vec(mcore::detail::expression<double, E> const& e)
			: data_(e) {}

		template <typename E>
		vec& operator=(mcore::detail::expression<double, E> const& e) {
			data_ = e;
			return *this;
		}

		size_t dim() const noexcept { return data_.size(); }
		size_t size() const noexcept { return data_.size(); }

		/** @brief Returns copy of this
		 */
		vec copy() const { return data_.copy(); }

		/** @brief Create a copy of a vector from another
		 */
		void assign(vec const& another) { data_.assign(another.data_); }

		/** @brief Change size of an object
		 */
		void resize(size_t new_size) { data_.resize(new_size); }

		/** @brief Clears the vector
		 *
		 * The operation sets all the element in the vector equals to zero
		 */
		void to_zero() noexcept {
			std::memset(data_.data(), 0, sizeof(double) * data_.size());
		}

		/** @brief Returns elements of the vector
		 *
		 */
		container_type release() noexcept {
			return std::move(data_);
		}

		double operator()(size_t index) const noexcept {
			assert(index < data_.size());
			return data_[index];
		}

		double& operator()(size_t index) noexcept {
			assert(index < data_.size());
			return data_[index];
		}

		double operator[](size_t index) const noexcept {
			assert(index < data_.size());
			return data_[index];
		}

		double& operator[](size_t index) noexcept {
			assert(index < data_.size());
			return data_[index];
		}

		container_type const& data() const noexcept {
			return data_;
		}

		/** @brief Increment operation
		 */
		void operator+=(vec const& other) noexcept {
			// Sizes of this and other are checked by sequence object
			data_.increment_unsafe(other.data_);
		}

		void operator-=(vec const& other) noexcept {
			// Sizes are validated ba a sequence object.
			data_.decrement_unsafe(other.data_);
		}

		void operator*=(double const val) noexcept {
			data_.mul_assign(val);
		}

		void operator/=(double const val) noexcept {
			data_.div_assign(val);
		}

		friend mcore::detail::subtraction<container_type, container_type>
		operator-(vec const& x, vec const& y) {
			assert(x.dim() == y.dim());
			return mcore::detail::sub<
				container_type,
				container_type,
				mcore::detail::indifferent_size_policy>(x.data_, y.data_);
		}

		friend mcore::detail::addition<container_type, container_type>
		operator+(vec const& x, vec const& y) {
			assert(x.dim() == y.dim());
			return mcore::detail::add<
				container_type,
				container_type,
				mcore::detail::indifferent_size_policy>(x.data_, y.data_);
		}

		template <typename E>
		mcore::detail::addition<
				container_type,
				mcore::detail::expression<double, E>>
		friend operator+(vec const& x, mcore::detail::expression<double, E> const& e) {
			assert(x.dim() == e.size());
			return mcore::detail::add<
				container_type,
				mcore::detail::expression<double, E>,
				mcore::detail::indifferent_size_policy>(x.data_, e);
		}

		template <typename E>
		mcore::detail::addition<
				container_type,
				mcore::detail::expression<double, E>>
		friend operator+(mcore::detail::expression<double, E> const& e, vec const& x) {
			assert(x.dim() == e.size());
			return mcore::detail::add<
				container_type,
				mcore::detail::expression<double, E>,
				mcore::detail::indifferent_size_policy>(x.data_, e);
		}

	private:
		vec(container_type&& ncs) noexcept
			: data_(std::move(ncs)) {}

	private:
		container_type data_;
	};

/** @brief Matrix implementation
 */
class mat {
public:
	typedef double value_type;

private:
	typedef ::mcore::detail::sequence<value_type> container_type;

public:
	/** @brief Creates matrix of required sizes
	 *
	 * @param rows number of rows in the matrix
	 * @aram cols number of columns in the matrix
	 *
	 */
	mat(size_t rows, size_t cols)
		: data_(rows * cols), rows_(rows), cols_(cols) {}

	double operator()(size_t r, size_t c) const noexcept {
		assert(r < rows_);
		assert(c < cols_);
		return GET_ELEMENT(data_, rows_, cols_, r, c);
	}

	double& operator()(size_t r, size_t c) noexcept {
		assert(r < rows_);
		assert(c < cols_);
		return GET_ELEMENT(data_, rows_, cols_, r, c);
	}

	mat copy() const;

	size_t rows() const noexcept { return rows_; }
	size_t cols() const noexcept { return cols_; }

	container_type const& data() const noexcept { return data_; }

private:
	mat(size_t rows, size_t cols, container_type&& data) noexcept
		: data_(std::move(data)), rows_(rows), cols_(cols) {}

private:
	container_type data_;
	size_t rows_{0};
	size_t cols_{0};
};


	/** @brief Compres two vectors
	 *
	 * @param x   first vector to compare
	 * @param y   second vector to compare
	 * @param tol required tolerance of the operation
	 *
	 * x and y are considered to be equal if they have equal dimensions
	 * and theres data are more or less the same
	 */
	bool eq(vec const& x, vec const& y, double tol=1.e-12) noexcept;

	/** @brief Compres two matrics
	 *
	 * @param x   first matrix
	 * @param y   second matrix
	 * @param tol required tolerance of the operation
	 */
	bool eq(mat const& x, mat const& y, double tol=1.e-12) noexcept;


namespace detail {

class mat_view {
public:
	typedef double element_type;
	friend class mcore::linalg::leq_solver;

public:
	explicit mat_view(size_t nrows) noexcept
		: nrows_(nrows) {}

	element_type& operator()(size_t r, size_t c) noexcept {
		assert(r < nrows_);
		assert(c < nrows_);
		assert(data_);

		return GET_ELEMENT(data_, nrows_, nrows_, r, c);
	}

	element_type const& operator()(size_t r, size_t c) const noexcept {
		return const_cast<element_type const&>(
		const_cast<mat_view*>(this)->operator()(r, c));
	}

private:
	void copy_from(double const* ptr) noexcept {
		assert(data_);
		std::memcpy(data_, ptr, sizeof(double) * nrows_ * nrows_);
	}

private:
	size_t nrows_;
	element_type* data_{nullptr};
};


template <typename T>
class vec_view {
public:
	typedef T element_type;
	friend class mcore::linalg::leq_solver;

public:
	explicit vec_view(size_t dim) noexcept
		: dimension_(dim) {}

	element_type& operator()(size_t index) noexcept {
		assert(index < dimension_);
		return data_[index];
	}

	element_type const& operator()(size_t index) const noexcept {
		return const_cast<element_type const&>(
		const_cast<vec_view*>(this)->operator()(index));
	}

	element_type& operator[](size_t index) noexcept {
		assert(index < dimension_);
		return data_[index];
	}

	element_type const& operator[](size_t index) const noexcept {
		return const_cast<element_type const&>(
		const_cast<vec_view*>(this)->operator[](index));
	}

private:
	void copy_from(double const* ptr) noexcept {
		assert(data_);
		std::memcpy(data_, ptr, sizeof(double) * dimension_);
	}

private:
	size_t dimension_;
	element_type* data_{nullptr};
};

} // namespace detail


/** @brief The class solve system of linear equations
 */
class leq_solver {
	typedef double element_type;

	leq_solver(leq_solver&) = delete;
	leq_solver& operator=(leq_solver&) = delete;

	static size_t get_memory_size(size_t degree) noexcept;

public:
	explicit leq_solver(size_t degree);

	vec solve(mat const& x, vec const& y);
	void solve(mat const& x, vec const& y, vec& rhs);

private:
	void init_permutation() noexcept;
	void triangulate(mat const& x, vec const& y) noexcept;

private:
	size_t dim_;
	detail::mat_view A_;
	detail::vec_view<double> rhs_;
# ifdef WITH_LAPACK
	detail::vec_view<int> permutation_;
# else
	detail::vec_view<size_t> permutation_;
# endif
	std::unique_ptr<uint8_t[]> data_;
};

/** @brief Solves a system of linear equations
 */
inline vec solve(mat const& x, vec const& y) {
	return leq_solver(y.dim()).solve(x, y);
}

// =============================================================================
// Low-level matrix operation

/*! \brief Solve Ax = y where A is a tridiagonal matrix
 *
 * @param d0 subdiagonal elements of A [1..dim - 1]
 * @param d1 diagonal of A [0..dim - 1]
 * @param d2 superdiagonal of A [0..dim - 2]
 * @param x before call holds an input (rhs), after the call -- an output values
 *          during the function all the elements of the array are addressed
 * @param dim dimensionality of the problem
 *
 * The function does not check for diagonal dominance, so the result is not
 * guaranteed to be stable.
 */
void solve_tridiagonal(
	double* d0,
	double* d1,
	double* d2,
	double* x,
	const size_t dim) noexcept;

}} // namespace linalg // namespace mcore

# undef GET_ELEMENT