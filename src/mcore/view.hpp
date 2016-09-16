# pragma once

# include <cassert>
# include <cstring>
# include <cstddef>
# include <initializer_list>

# include "config.hpp"

namespace mcore { namespace linalg {

class leq_solver;

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
class transposed_view {
public:
	typedef typename T::value_type value_type;

public:
	explicit transposed_view(T const& matrix) noexcept
		: matrix_(matrix) {}

	size_t rows() const noexcept { return matrix_.cols(); }
	size_t cols() const noexcept { return matrix_.rows(); }

	value_type operator()(size_t r, size_t c) const {
		return matrix_(c, r);
	}

private:
	T const& matrix_;
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

}}} // namespace linalg // namespace mcore
