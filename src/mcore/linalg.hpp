# pragma once

# include <cassert>
# include <memory>


namespace mcore { namespace linalg {

	template <typename T>
	struct expression {
		double operator()(size_t index) const noexcept { return static_cast<T const&>(*this)(index); }
		double operator()(size_t r, size_t c) const noexcept { return static_cast<T const&>(*this)(r, c); }

		size_t rows() const noexcept { return static_cast<T const&>(*this).rows(); }
		size_t cols() const noexcept { return static_cast<T const&>(*this).cols(); }
	};


	template <typename X, typename Y>
	class multiplication : public expression<multiplication<X, Y>> {
	public:
		multiplication(X const& x, Y const& y);

		multiplication(multiplication const& other)
			: x_(other.x_), y_(other.y_) {}

		double operator()(size_t index) const noexcept {
			double r = 0;
			for (size_t j = 0; j < x_.cols(); ++j)
				r += x_(index, j) * y_(j);
			return r;
		}

		double operator()(size_t r, size_t c) const noexcept {
			double rv = 0;
			for (size_t j = 0; j < x_.cols(); ++j)
				rv += x_(r, j) * y_(j, c);
			return rv;
		}

		size_t rows() const noexcept { return x_.rows(); }
		size_t cols() const noexcept { return y_.cols(); }

	private:
		X const& x_;
		Y const& y_;
	};

	template <typename X>
	class multiplication_1 : public expression<multiplication_1<X>> {
	public:
		multiplication_1(double k, X const& x) noexcept
			: k_(k), x_(x) {}

		double operator()(size_t index) const noexcept {
			return k_ * x_(index);
		}

		double operator()(size_t r, size_t c) const noexcept {
			return k_ * x_(r, c);
		}

		size_t rows() const noexcept { return x_.rows(); }

		size_t cols() const noexcept { return x_.cols(); }

	private:
		double k_;
		X const& x_;
	};

	template <typename T>
	class t_expression : public expression<t_expression<T>> {
	public:
		explicit t_expression(T const& t) noexcept
			: arg_(t) {}

		double operator()(size_t r, size_t c) const noexcept {
			return arg_(c, r);
		}

		size_t rows() const noexcept { return arg_.cols(); }
		size_t cols() const noexcept { return arg_.rows(); }

	private:
		T const& arg_;
	};


	class vec {
	public:
		typedef double value_type;

		vec() noexcept
			: data_{nullptr}, dim_{0} {}

		explicit vec(size_t dim)
			: data_{new double[dim]}, dim_{dim} {}

		vec(vec&& other) noexcept
				: data_(std::move(other.data_)), dim_(other.dim_) {
			other.dim_ = 0;
		}

		template <typename T>
		vec& operator=(expression<T>&& e) {
			if (&e != this) {
				// Do we need to reallocate memory?
				if (dim_ < e.rows()) {
					data_.reset(new double[e.cols()]);
					dim_ = e.cols();
				}
				for (size_t i = 0; i < dim_; ++i)
					data_[i] = e(i);
			}
			return *this;
		}

		template <typename T>
		vec(expression<T>&& e)
				: data_{new double[e.rows()]}, dim_{e.rows()} {
			for (size_t i = 0; i < dim_; ++i)
				data_[i] = e(i);
		}

		/** @brief Creates copy of the vector
		 */
		vec copy() const;

		/** @brief Clears the vector
		 *
		 * The operation sets all the elements in the vector to zero.
		 */
		void clear();

		value_type const& operator()(size_t index) const noexcept {
			assert(index < dim_);
			return data_[index];
		}

		value_type& operator()(size_t index) noexcept {
			return const_cast<value_type&>(
				const_cast<vec const*>(this)->operator()(index));
		}

		value_type const& at(size_t index) const {
			if (index < dim_)
				return data_[index];
			else
				// TODO :: fix it
				throw 1;
		}

		value_type& at(size_t index) {
			return const_cast<value_type&>(
				const_cast<vec const*>(this)->at(index));
		}

		size_t rows() const noexcept {
			return dim_;
		}

	private:
		vec(vec&) = delete;
		vec& operator=(vec&) = delete;

	private:
		std::unique_ptr<value_type[]> data_;
		size_t dim_;
	};

	class mat {
	public:
		typedef double value_type;

		mat() noexcept
			: data_{nullptr}, row_{0}, col_{0} {}

		mat(size_t rs, size_t cs)
			: data_{new double[rs * cs]}, row_{rs}, col_{cs} {}

		mat(mat&& other) noexcept
				: data_(std::move(other.data_)), row_{other.row_}, col_{other.col_} {
			other.row_ = 0;
			other.col_ = 0;
		}

		template <typename T>
		mat(expression<T>&& e)
				: data_{new double[e.cols() * e.rows()]}, row_{e.rows()}, col_{e.cols()} {
			for (size_t i = 0; i < row_; ++i)
				for (size_t j = 0; j < col_; ++j)
					data_[i * col_ + j] = e(i, j);
		}

		value_type const& operator()(size_t r, size_t c) const noexcept {
			assert(r < row_);
			assert(c < col_);
			return data_[r * col_ + c];
		}

		/** @brief Creates copy of the matrix
		 */
		mat copy() const;

		value_type& operator()(size_t r, size_t c) noexcept {
			return const_cast<value_type&>(
				const_cast<mat const*>(this)->operator()(r, c));
		}

		value_type const& at(size_t r, size_t c) const {
			if (r < row_ && c < col_)
				return data_[r * col_ + c];
			else
				// TODO :: fix it
				throw 1;
		}

		value_type& at(size_t r, size_t c) {
			return const_cast<value_type&>(
				const_cast<mat const*>(this)->at(r, c));
		}

		size_t rows() const noexcept { return row_; }
		size_t cols() const noexcept { return col_; }

	private:
		mat(mat&) = delete;
		mat& operator=(mat&) = delete;

	private:
		std::unique_ptr<value_type[]> data_;
		size_t row_;
		size_t col_;
	};

	template <>
	inline multiplication<mat, vec>::multiplication(mat const& x, vec const& y)
			: x_(x), y_(y) {
		assert(x.cols() == y.rows());
	}

	template <>
	inline multiplication<mat, mat>::multiplication(mat const& x, mat const& y)
			: x_(x), y_(y) {
		assert(x.cols() == y.rows());
	}

	inline t_expression<mat> transposed(mat const& x) noexcept {
		return t_expression<mat>(x);
	}


	// Multiplication operations

	inline multiplication<mat, mat> operator*(mat const& x, mat const& y) noexcept {
		return multiplication<mat, mat>(x, y);
	}

	inline multiplication<mat, vec> operator*(mat const& x, vec const& y) noexcept {
		return multiplication<mat, vec>(x, y);
	}

	template <typename V>
	multiplication_1<V> operator*(double k, V const& x) noexcept {
		return multiplication_1<V>(k, x);
	}

	template <typename V>
	multiplication_1<V> operator*(V const& v, double k) noexcept {
		return multiplication_1<V>(k, v);
	}

	template <typename V>
	multiplication_1<V> operator/(V const& v, double k) noexcept {
		return multiplication_1<V>(v, 1. / k);
	}

	// Arithmetic operations

	// Comparison operator

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

	// Solve operation
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
