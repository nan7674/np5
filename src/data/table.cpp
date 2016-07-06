# include "table.hpp"

# include <string>
# include <cctype>
# include <functional>
# include <sstream>
# include <iostream>

namespace {

	void trim(std::string& s) noexcept {
		size_t const nchars = s.length();
		size_t pos0 = 0;
		for (; pos0 < nchars && std::isspace(s[pos0]); ++pos0);

		size_t pos1 = nchars - 1;
		for (; pos0 < nchars && std::isspace(s[pos1]); --pos1);
		s = s.substr(pos0, pos1 - pos0 + 1);
	}

	int parse_int(std::string const& s) {
		return atoi(s.c_str());
	}

	std::tuple<std::string, bool> get_line(std::istream& stream, char ch) {
		std::string line;
		std::getline(stream, line, ch);
		bool const is_eol = line.empty();
		trim(line);
		return std::make_tuple(line, is_eol);
	}


	void get_row(std::string const& line, std::vector<data::value>& container) {
		std::istringstream stream(line);
		do {
			auto p = get_line(stream, ' ');
			if (std::get<1>(p))
				break;
			container.emplace_back(data::value{});
			std::string const& token = std::get<0>(p);
			if (!token.empty()) {
				int const i = parse_int(token);
				container.back().assign(i);
			}
		} while (stream);
	}
}


void data::table::load_from(std::istream& iss, header const& hdr) {
}


void data::table::load_from(std::istream& iss) {
	{
		// The very first row will be handled separately
		std::string line;
		while (iss && line.empty()) {
			auto p = get_line(iss, '\n');
			std::get<0>(p).swap(line);
		}

		if (!iss)
			return;

		std::vector<value> row;
		get_row(line, row);
		row.shrink_to_fit();
		rows_.push_back(std::move(row));
	}
	size_t const ncols = rows_.back().size();


	while (iss) {
		auto p = get_line(iss, '\n');
		if (std::get<1>(p))
			continue;

		std::string const& line = std::get<0>(p);
		std::vector<value> row;
		row.reserve(ncols);
		get_row(line, row);

		rows_.emplace_back(std::move(row));
	}
}


std::ostream& data::operator<<(std::ostream& stream, table const& t) {
	for (auto const& r : t.rows_) {
		for (auto const& v: r) {
			if (v.is_defined())
				stream << v.get<int>() << ", ";
			else
				stream << "(nan)" << ", ";
		}
		stream << '\n';
	}
	return stream;
}


# define WITH_UT
# ifdef WITH_UT

# include <cassert>

namespace test {
	void test1() {
		{
			std::string a = "   123    ";
			trim(a);
			assert(a == "123");
		}

		{
			std::string a;
			trim(a);
			assert(a.empty());
		}
	}

	void test2() {
		std::istringstream iss(
		"season1 season2 season3\n"
		"1 2 3\n"
		"4 5 6\n"
		"1  1\n");

		data::table t;
		t.load_from(iss);

		std::cout << t << std::endl;

		data::column col = t.get_column(0);
		std::cout << col.sum<int>(0, 3) << std::endl;
	}

	void test3() {
		std::string a = "1 2 3";
		std::vector<data::value> data;
		get_row(a, data);
	}


	void test4() {
		std::string c;
		std::istringstream l(c);
		size_t n = 0;
		while (l) {
			std::string t;
			std::getline(l, t);
			std::cout << t.size() << ' ' << t.length() << std::endl;
			std::cout << "num " << n++ << std::endl;
		}
	}
}

int main() {
	test::test2();
//	test::test1();
//	test::test3();
//	test::test4();
	return 0;
}

# endif
