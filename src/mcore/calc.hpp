# pragma once

# include <initializer_list>

# include "sequence.hpp"

namespace mcore { namespace calc {

template <typename V, typename T>
T poly_eval(V const& v, T const x) noexcept {
	size_t const sz = v.size();
	T s = v[sz - 1];
	for (size_t i = sz - 2; i < sz; --i)
		s = v[i] + s * x;
	return s;
}

class polynom {
public:
	typedef double value_type;
	typedef ::mcore::detail::sequence<value_type> container_type;

public:
	/** @brief Creates empty polynom
	 */
	polynom() {}

	/** @brief Creates a polynom of a required degree.
	 *
	 * @param degree degree of the polynom.
	 *
	 * All the coefficients inthe polynom ae set to zero.
	 */
	explicit polynom(size_t degree)
		: coeffs_(degree, 0) {}

	polynom(std::initializer_list<double> L)
		: coeffs_(L) {}

	polynom(container_type&& c) noexcept
			: coeffs_(std::move(c)) {}

	/** @brief Creates a copy of this
	 *
	 * @return a copy of this
	 */
	polynom copy() const { return coeffs_.copy(); }

	template <typename E>
	polynom(::mcore::detail::expression<double, E> const& e)
		: coeffs_(e) {}

	size_t degree() const noexcept {
		return coeffs_.size() == 0 ? 0 : coeffs_.size() - 1;
	}

	void swap(polynom& other) noexcept {
		std::swap(coeffs_, other.coeffs_);
	}

	/** @brief Returns number of a coefficients in the polynom
	 */
	size_t size() const noexcept { return coeffs_.size(); }

	double operator[](size_t index) const noexcept {
		return index < coeffs_.size() ? coeffs_[index] : 0;
	}

	double& operator[](size_t index);

	// TODO :: fix result of the polynomial degree
	friend ::mcore::detail::addition<container_type, container_type>
	operator+(polynom const& p1, polynom const& p2) {
		return ::mcore::detail::add<
			container_type,
			container_type,
			::mcore::detail::indifferent_size_policy>(p1.coeffs_, p2.coeffs_);
	}

	// TODO :: fix result of the degree
	friend ::mcore::detail::subtraction<container_type, container_type>
	operator-(polynom const& p1, polynom const& p2) {
		return ::mcore::detail::sub<
			container_type,
			container_type,
			::mcore::detail::indifferent_size_policy>(p1.coeffs_, p2.coeffs_);
	}

	friend ::mcore::detail::multiplication<container_type>
	operator*(polynom const& p1, double k) {
		return ::mcore::detail::multiply(k, p1.coeffs_);
	}

	friend ::mcore::detail::multiplication<container_type>
	operator*(double k, polynom const& p1) {
		return ::mcore::detail::multiply(k, p1.coeffs_);
	}

	double operator()(double x) const noexcept {
		return poly_eval(coeffs_, x);
	}

private:
	container_type coeffs_;
};


/** @brief Shortcut for square
 */
inline double sqr(double x) noexcept { return x * x; }

}} // namespace calc //namespace mcore
