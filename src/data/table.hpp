# pragma once

# include <istream>
# include <ostream>
# include <vector>

namespace data {


class value {
	union data_size {
		int i_;
		double d_;
	};

public:
	value() = default;

	template <typename T>
	void assign(T const v) noexcept {
		T* ptr = (T*) memory_;
		*ptr = v;

		defined_ = true;
	}


	template <typename T>
	T get() const noexcept {
		return *(T*)memory_;
	}

	bool is_defined() const noexcept {
		return defined_;
	}

private:
	char memory_[sizeof(data_size)];
	bool defined_{false};
};


class table;

std::ostream& operator<<(std::ostream&, table const&);


class column {
	friend class table;

	column& operator=(column&) = delete;

private:
	explicit column(table const& t, size_t c) noexcept 
		: table_(t), index_(c) {}

public:
	template <typename T>
	T get(size_t const index) const;

	template <typename T>
	T sum(size_t from, size_t to) const;

private:
	table const& table_;
	size_t const index_;
};


class header {
};


class table {
	table(table const&) = delete;
	table& operator=(table&) = delete;

	typedef std::vector<value> row_type;

	friend std::ostream& operator<<(std::ostream&, table const&);

public:
	table() = default;

	void load_from(std::istream&);

	void load_from(std::istream&, header const& hdr);

	size_t num_rows() const noexcept {
		return rows_.size();
	}

	template <typename T>
	T get(size_t r, size_t c) const {
		return rows_.at(r).at(c).get<T>();
	}

	column get_column(size_t const index) const {
		return column(*this, index);
	}

private:
	std::vector<row_type> rows_;
};


template <typename T>
T column::get(size_t const index) const {
	return table_.get<T>(index, index_);
}


template <typename T>
T column::sum(size_t const from, size_t const to) const {
	T result = 0;
	for (size_t i = from; i < to + 1; ++i)
		result += table_.get<T>(i, index_);

	return result;
}


} // data
