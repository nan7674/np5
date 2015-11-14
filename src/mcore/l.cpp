# include "linalg.hpp"

# include <iostream>
# include <type_traits>

namespace m = mcore::linalg;

template <typename T>
bool check(T&& val) {
	return std::is_arithmetic<typename std::remove_reference<T>::type>::value;
}

int main() {

	m::vec p(2);
	p(0) = 3;
	p(1) = 4;

	m::mat q(2, 2);
	q(0, 0) = 2;
	q(1, 0) = 0;
	q(0, 1) = 0;
	q(1, 1) = 6;
	std::cout << "mat " << (void*)&q << std::endl;

	{
		std::cout << "multiplication " << std::endl;
		m::vec r = q * p;
		std::cout << r(0) << std::endl;
		std::cout << r(1) << std::endl;
	}

	{
		m::vec r = 3 * p;
		std::cout << r(0) << std::endl;
		std::cout << r(1) << std::endl;
	}

	{
		m::vec r = p * 3;
		std::cout << r(0) << std::endl;
		std::cout << r(1) << std::endl;
	}

	{
		m::mat r = 3 * q;
		std::cout << r(0, 0) << ' ' << r(0, 1) << std::endl;
		std::cout << r(1, 0) << ' ' << r(1, 1) << std::endl;
	}

	{
		m::mat r = q * 5;
		std::cout << r(0, 0) << ' ' << r(0, 1) << std::endl;
		std::cout << r(1, 0) << ' ' << r(1, 1) << std::endl;
	}

	{
		m::mat m1(2, 3);
		m::mat m2(3, 2);

		m1(0, 0) = 1; m1(0, 1) = 2; m1(0, 2) = 3;
		m1(1, 0) = 4; m1(1, 1) = 5; m1(1, 2) = 6;

		m2(0, 0) =  7; m2(0, 1) = 8;
		m2(1, 0) =  9; m2(1, 1) = 10;
		m2(2, 0) = 11; m2(2, 1) = 12;

		//                      /   7   8
		//   / 1   2   3 \     |
		//  |             | *  |    9  10
		//   \ 4   5   6 /     |
		//                      \  11  12
		//


		m::mat m3 = m1 * m2;

		std::cout << m3(0, 0) << ' ' << (1 * 7 + 2 * 9 + 3 * 11) << std::endl;
		std::cout << m3(0, 1) << ' ' << (1 * 8 + 2 * 10 + 3 * 12) << std::endl;
		std::cout << m3(1, 0) << ' ' << (4 * 7 + 5 * 9 + 6 * 11) << std::endl;
		std::cout << m3(1, 1) << ' ' << (4 * 8 + 5 * 10 + 6 * 12) << std::endl;
	}

	{
		m::vec r = p.copy();
		std::cout << std::boolalpha << m::eq(r, p) << std::endl;
	}

	{
		m::mat r = q.copy();
		std::cout << std::boolalpha << m::eq(r, q) << std::endl;
	}

	{
		m::mat A(2, 2);
		A(0, 0) = 1; A(0, 1) = -1;
		A(1, 0) = 2; A(1, 1) = 3;

		m::vec b(2);
		b(0) = 1;
		b(1) = 3;

		m::vec x = m::solve(A, b);
		std::cout << x(0) << ' ' << 1.2 << std::endl;
		std::cout << x(1) << ' ' << 0.2 << std::endl;
	}

	return 0;
}
