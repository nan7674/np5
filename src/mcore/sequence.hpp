# pragma once

# include <memory>
# include <type_traits>
# include <stdexcept>
# include <initializer_list>
# include <algorithm>
# include <cstring>
# include <cassert>


namespace mcore { namespace detail {

template <typename T>
class data_view {
protected:
	data_view(T* initial, size_t count) noexcept
		: data_(initial), size_(count) {}

public:
	size_t size() const noexcept { return size_; }

	T& operator[](size_t index) noexcept {
		assert(index < size_);
		return data_[index];
	}

	T const& operator[](size_t index) const noexcept {
		return const_cast<T const&>(const_cast<data_view*>(this)->operator[](index));
	}

	T const* begin() const noexcept { return data_; }
	T const* end() const noexcept { return data_ + size_; }

private:
	T* data_;
	size_t size_;
};

template <typename T, typename X>
struct expression {
	typedef T value_type;

	size_t size() const noexcept {
		return static_cast<X const*>(this)->size();
	}

	T* get_data() const {
		return static_cast<X const*>(this)->get_data();
	}

	value_type operator[](size_t idx) const noexcept {
		return static_cast<X const*>(this)->operator[](idx);
	}
};


template <typename C>
class multiplication : public expression<typename C::value_type, multiplication<C>> {
public:
	typedef typename C::value_type value_type;

	multiplication(value_type coeff, C const& c) noexcept
		: coeff_(coeff), container_(c) {}

	size_t size() const noexcept {
		return container_.size();
	}

	value_type operator[](size_t idx) const noexcept {
		return container_[idx] * coeff_;
	}

	value_type* get_data() const {
		size_t const sz = container_.size();
		value_type* out = new value_type[sz];
		for (size_t i = 0; i < sz; ++i)
			out[i] = coeff_ * container_[i];
		return out;
	}

private:
	value_type coeff_;
	C const& container_;
};


template <typename A1, typename A2>
struct check_number_presence {
	enum {
		value =
			std::is_arithmetic<A1>::value ||
			std::is_arithmetic<A2>::value
	};
};


template <typename A1, typename A2>
struct arithmetic_first_type {
	typedef typename A2::value_type value_type;
	typedef A2 argument_type;

	static multiplication<A2> build(A1 const& a1, A2 const& a2) {
		return multiplication<A2>(a1, a2);
	}
};


template <typename A1, typename A2>
struct arithmetic_second_type {
	typedef typename A1::value_type value_type;
	typedef A1 argument_type;

	static multiplication<A1> build(A1 const& a1, A2 const& a2) {
		return multiplication<A1>(a2, a1);
	}
};

template <typename A1, typename A2>
struct scalar_multiplication_builder {
	typedef typename std::conditional<
		std::is_arithmetic<A1>::value,
		arithmetic_first_type<A1, A2>,
		arithmetic_second_type<A1, A2>
	>::type builder_type;

	typedef typename builder_type::argument_type argument_type;

	static multiplication<argument_type>
	build(A1 const& a1, A2 const& a2) {
		return builder_type::build(a1, a2);
	}
};


template <typename A1, typename A2>
multiplication<typename scalar_multiplication_builder<A1, A2>::argument_type>
multiply(A1 const& a1, A2 const& a2) {
	typedef scalar_multiplication_builder<A1, A2> builder_type;
	return builder_type::build(a1, a2);
}


template <typename A1, typename A2>
multiplication<typename scalar_multiplication_builder<A1, A2>::argument_type>
operator*(A1 const& a1, A2 const& a2) {
	typedef scalar_multiplication_builder<A1, A2> builder_type;
	return builder_type::build(a1, a2);
}


template <typename A1, typename A2>
class addition : public expression<typename A1::value_type, addition<A1, A2>> {
public:
	typedef typename A1::value_type value_type;

	addition(A1 const& a1, A2 const& a2) noexcept
		: arg1_(a1), arg2_(a2)
	{
		size_t const s1 = arg1_.size();
		size_t const s2 = arg2_.size();

		size_ = s1 > s2 ? s1 : s2;
	}

	size_t size() const noexcept { return size_; }

	value_type* get_data() const {
		size_t const s1 = arg1_.size();
		size_t const s2 = arg2_.size();
		value_type* out = nullptr;
		if (s1 < s2) {
			out = new value_type[s2];
			for (size_t i = 0; i < s1; ++i)
				out[i] = arg1_[i] + arg2_[i];
			for (size_t i = s1; i < s2; ++i)
				out[i] = arg2_[i];
		} else {
			out = new value_type[s1];
			for (size_t i = 0; i < s2; ++i)
				out[i] = arg1_[i] + arg2_[i];
			for (size_t i = s2; i < s1; ++i)
				out[i] = arg1_[i];
		}

		return out;
	}

	value_type operator[](size_t idx) const noexcept {
		return
			(idx < arg1_.size() ? arg1_[idx] : 0) +
			(idx < arg2_.size() ? arg2_[idx] : 0);
	}

private:
	A1 const& arg1_;
	A2 const& arg2_;
	size_t size_;
};


template <typename A1, typename A2>
class subtraction : public expression<typename A1::value_type, subtraction<A1, A2>> {
public:
	typedef typename A1::value_type value_type;

	subtraction(A1 const& a1, A2 const& a2) noexcept
		: arg1_(a1), arg2_(a2)
	{
		size_t const s1 = arg1_.size();
		size_t const s2 = arg2_.size();

		size_ = s1 > s2 ? s1 : s2;
	}

	size_t size() const noexcept { return size_; }

	value_type* get_data() const {
		size_t const s1 = arg1_.size();
		size_t const s2 = arg2_.size();
		value_type* out = nullptr;
		if (s1 < s2) {
			out = new value_type[s2];
			for (size_t i = 0; i < s1; ++i)
				out[i] = arg1_[i] - arg2_[i];
			for (size_t i = s1; i < s2; ++i)
				out[i] = -arg2_[i];
		} else {
			out = new value_type[s1];
			for (size_t i = 0; i < s2; ++i)
				out[i] = arg1_[i] - arg2_[i];
			for (size_t i = s2; i < s1; ++i)
				out[i] = arg1_[i];
		}

		return out;
	}

	value_type operator[](size_t idx) const noexcept {
		return
			(idx < arg1_.size() ? arg1_[idx] : 0) -
			(idx < arg2_.size() ? arg2_[idx] : 0);
	}

private:
	A1 const& arg1_;
	A2 const& arg2_;
	size_t size_;
};


template <typename A1, typename A2, typename P>
addition<A1, A2> add(A1 const& a1, A2 const& a2) {
	P{}.template validate<A1, A2>(a1, a2);
	return addition<A1, A2>(a1, a2);
}


template <typename A1, typename A2, typename P>
subtraction<A1, A2> sub(A1 const& a1, A2 const& a2) {
	P{}.template validate<A1, A2>(a1, a2);
	return subtraction<A1, A2>(a1, a2);
}


struct equal_size_policy {
	template <typename A1, typename A2>
	void validate(A1 const& a1, A2 const& a2) const {
		size_t const s1 = a1.size();
		size_t const s2 = a2.size();

		if (s1 != s2)
			throw std::runtime_error("Argument have different sizes.");
	}
};


struct indifferent_size_policy {
	template <typename A1, typename A2>
	void validate(A1 const& a1, A2 const& a2) const {}
};


template <typename T>
class sequence {
public:
	typedef size_t size_type;
	typedef T value_type;

	sequence(sequence&) = delete;
	sequence& operator= (sequence&) = delete;

public:
	/** @brief Creates empty sequence
	 */
	sequence() noexcept
		: size_{0}, data_{nullptr} {}

	/** @brief Creates sequence of a given length
	 *
	 * @param sz size (length) of the sequence
	 */
	explicit sequence(size_type sz)
		: size_{sz}, data_(new value_type[sz]) {}

	/** @brief Creates sequence of a given length.
	 *
	 * @param sz size (length) of the sequence
	 * @param val value of an element in the sequence
	 *
	 * All the members in the sequence are set to
	 * the value val.
	 */
	sequence(size_type sz, T val)
		: size_{sz}, data_(new value_type[sz])
	{
		std::fill_n(data_.get(), sz, val);
	}

	/** @brief Given an initializer_list creates a sequence from it
	 */
	template <typename X>
	sequence(std::initializer_list<X> l) : size_(l.size()) {
		data_.reset(new value_type[size_]);
		std::copy(l.begin(), l.end(), data_.get());
	}

	sequence(sequence&& other) noexcept
		: size_(other.size_), data_(std::move(other.data_)) {}

	sequence& operator=(sequence&& other) noexcept {
		if (this != &other) {
			size_ = other.size_;
			other.size_ = 0;
			data_ = std::move(other.data_);
		}
		return *this;
	}

	/** @brief Creates a copy of this
	 */
	sequence copy() const {
		sequence this_copy(size_);
		std::memcpy(this_copy.data_.get(), data_.get(), sizeof(T) * size_);
		return this_copy;
	}

	/** @brief Explicit copy-constructor
	 */
	void assign(sequence const& other) {
		if (size_ != other.size_) {
			data_.reset(new T[other.size_]);
			size_ = other.size_;
		}
		std::memcpy(data_.get(), other.data_.get(), sizeof(T) * size_);
	}

	/** @brief Creates sequence from an expression
	 */
	template <typename X>
	sequence(expression<T, X> const& e) : size_{e.size()} {
		data_.reset(e.get_data());
	}

	template <typename X>
	sequence& operator=(expression<T, X> const& e) {
		// TODO :: we dont need to do memory reallocation
		if (size_ != e.size()) {
			size_ = e.size();
			data_.reset(e.get_data());
		} else {
			for (size_t i = 0; i < size_; ++i)
				data_[i] = e[i];
		}
		return *this;
	}

	/** @brief Returns size of the sequence
	 */
	size_t size() const noexcept {
		return size_;
	}

	T operator[](size_t idx) const noexcept {
		assert(idx < size_);
		return data_[idx];
	}

	T& operator[](size_t idx) noexcept {
		assert(idx < size_);
		return data_[idx];
	}

	/** @brief Returns pointer to data
	 */
	T const* data() const noexcept { return data_.get(); }
	T* data() noexcept { return data_.get(); }

	/** @brief Resize the sequence
	 *
	 * If the required size of the sequence is more then current
	 * all the data are copied into the new container. Every
	 * added element is assigned to zero
	 */
	void resize(size_t ns) {
		assert(ns != size_);
		value_type* nd = new value_type[ns];
		if (ns > size_) {
			size_t const nbytes = sizeof(T) * size_;
			std::memcpy(nd, data_.get(), nbytes);
			std::memset(nd + size_, 0, sizeof(T) * (ns - size_));
		} else {
			size_t const nbytes = sizeof(T) * ns;
			std::memcpy(nd, data_.get(), nbytes);
		}
		size_ = ns;
		data_.reset(nd);
	}

	/** @brief Increment operation (unsafe version)
	 */
	void increment_unsafe(sequence const& other) noexcept {
		assert(size_ == other.size_);
		T* p1 = data_.get();
		T const* p2 = other.data_.get();
		for (size_t i = 0; i < size_; ++i, ++p1, ++p2)
			*p1 += *p2;
	}

	/** @brief Decrement operation (unsafe version)
	 */
	void decrement_unsafe(sequence const& other) noexcept {
		assert(size_ == other.size_);
		T* p1 = data_.get();
		T const* p2 = other.data_.get();
		for (size_t i = 0; i < size_; ++i, ++p1, ++p2)
			*p1 -= *p2;
	}

	/** @brief Division assignment (unsafe version)
	 */
	void div_assign(T const val) noexcept {
		T* p = data_.get();
		for (size_t i = 0; i < size_; ++i, ++p)
			*p /= val;
	}

	/** @brief Multiplication assignment
	 */
	void mul_assign(T const val) noexcept {
		T* p = data_.get();
		for (size_t i = 0; i < size_; ++i, ++p)
			*p *= val;
	}


private:
	size_type size_;
	std::unique_ptr<value_type[]> data_;
};


}} // detail // mcore
