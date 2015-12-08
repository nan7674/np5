# pragma once

# include <cassert>
# include <memory>
# include <initializer_list>

# include "sequence.hpp"


namespace mcore { namespace linalg {

	class mat;
	class vec;

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

		/** @brief Change size of an object
		 */
		void resize(size_t new_size) { data_.resize(new_size); }

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
			return data_[r * cols_ + c];
		}

		double& operator()(size_t r, size_t c) noexcept {
			assert(r < rows_);
			assert(c < cols_);
			return data_[r * cols_ + c];
		}

		mat copy() const;

		size_t rows() const noexcept { return rows_; }
		size_t cols() const noexcept { return cols_; }

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

	/** @brief Solves system of linear equations
	 */
	vec solve(mat const& x, vec const& y);


// =============================================================================
// Low-level matrix operation

	/** @brief Cholesky decomposition of a symmetric 3-diagonal matrix
	 */
	void cholesky(
		double* const d0,
		double* const d1,
		double* const d2,
		size_t const n) noexcept;

	/** @brief Cholesky decomposition of a symmetric matrices
	 *
	 * @param d0 a main diagonal of the matrix
	 */
	void cholesky(
		double* const d0,
		double* const d1,
		size_t const n) noexcept;


	void solve_ldl(
		double const* const d0,
		double const* const d1,
		double const* const d2,
		double* const y,
		size_t const n) noexcept;


	void solve_ldl(
		double const* const d0,
		double const* const d1,
		double* const y,
		size_t const n) noexcept;

}} // namespace linalg // namespace mcore
