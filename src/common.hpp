# pragma once

# include <iostream>

# ifdef _WINDOWS
# define _ALLOW_KEYWORD_MACROS
# define noexcept throw()
# endif


#define MEMBER_DETECTOR(M)                                              \
template <typename Op>                                                  \
class has_member_##M {                                                  \
	                                                                    \
	typedef unsigned char  yep;                                         \
	typedef unsigned int   nope;                                        \
	                                                                    \
	template <typename U, U>                                            \
	struct check;                                                       \
	                                                                    \
	template <typename S>                                               \
	static yep test_member(check<double (S::*)(double), &S::M>*);       \
	                                                                    \
	template <typename S>                                               \
	static yep test_member(check<double (S::*)(double) const, &S::M>*); \
	                                                                    \
	template <typename S>                                               \
	static nope test_member(...);                                       \
	                                                                    \
public:                                                                 \
	static const bool value =                                           \
	    sizeof(decltype(test_member<Op>(0))) == sizeof(yep);            \
};

namespace np5 { namespace common {

	MEMBER_DETECTOR(diff);
	MEMBER_DETECTOR(diff2);

}} // 


# undef MEMBER_DETECTOR

namespace np5 {

	template <typename K>
	class singleton : public K {
	public:
		static singleton& instance() {
			static singleton S;
			return S;
		}

	private:
		singleton() : K() {}
	};

} // namespace np5