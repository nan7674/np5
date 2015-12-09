# include <Python.h>

# include <cassert>
# include <cstddef>
# include <cstring>
# include <cstdarg>
# include <sstream>
# include <vector>
# include <tuple>

# include "polynomial.hpp"


namespace {

	class wrapper_exception {};

	typedef Py_ssize_t (*get_size_function)(PyObject*);
	typedef PyObject* (*get_item_function)(PyObject*, Py_ssize_t);

	struct point {
		point() {}

		point(PyObject* a1, PyObject* a2)
			: x(PyFloat_AsDouble(a1)), y(PyFloat_AsDouble(a2)) {}

		point(PyObject* const object) noexcept
			: x(PyFloat_AsDouble(PyTuple_GetItem(object, 0))),
			  y(PyFloat_AsDouble(PyTuple_GetItem(object, 1))) {}

		point& operator=(PyObject* const object) {
			if (object) {
				x = PyFloat_AsDouble(PyTuple_GetItem(object, 0));
				y = PyFloat_AsDouble(PyTuple_GetItem(object, 1));
			}

			return *this;
		}

		void assign(PyObject* const a1, PyObject* const a2) {
			if (a1 && a2) {
				x = PyFloat_AsDouble(a1);
				y = PyFloat_AsDouble(a2);
			}
		}

		double x{0};
		double y{0};
	};

	template <PyObject* (*get_item)(PyObject*, Py_ssize_t)>
	class pyobject_iterator_1 {
	public:
		explicit pyobject_iterator_1(PyObject* object) noexcept
			: container_(object)
		{
			assert(container_ != nullptr);
			point_ = get_item(container_, 0);
		}

		pyobject_iterator_1(PyObject* object, size_t size) noexcept
			: container_(object), cursor_(size)
		{
			assert(container_ != nullptr);
		}

		bool operator!=(pyobject_iterator_1 const& it) const noexcept {
			assert(container_ == it.container_);
			return cursor_ != it.cursor_;
		}

		pyobject_iterator_1& operator++() noexcept {
			++cursor_;
			point_ = get_item(container_, cursor_);
			return *this;
		}

		point const* operator->() const { return &point_; }

	private:
		PyObject* const container_;
		size_t cursor_{0};
		point point_;
	};

	// This type of iterator incorporates idea of a ZIP iterator
	class pyobject_iterator_2 {
	public:
		// Ctor for initial position iterator
		pyobject_iterator_2(
				PyObject* first_sequence,
				get_item_function first_getter,
				PyObject* second_sequence,
				get_item_function second_getter) noexcept
			: s1_{first_sequence}, s2_{second_sequence},
			  get1_{first_getter}, get2_{second_getter},
			  cursor_{0}
		{
			assert(s1_ != nullptr);
			assert(s2_ != nullptr);
			assert(g1_ != nullptr);
			assert(g2_ != nullptr);

			point_.assign(get1_(s1_, 0), get2_(s2_, 0));
		}

		// Ctor for end position iterator
		pyobject_iterator_2(
				PyObject* first_sequence,
				PyObject* second_sequence,
				size_t size) noexcept
			: s1_{first_sequence}, s2_{second_sequence}, cursor_(size)
		{
			assert(s1_ != nullptr);
			assert(s2_ != nullptr);
		}

		bool operator!=(pyobject_iterator_2 const& it) const noexcept {
			assert(s1_ == it.s1_);
			assert(s2_ == it.s2_);
			return cursor_ != it.cursor_;
		}

		pyobject_iterator_2& operator++() noexcept {
			++cursor_;
			point_.assign(get1_(s1_, cursor_), get2_(s2_, cursor_));
			return *this;
		}

		point const* operator->() const noexcept { return &point_; }

	private:
		PyObject* s1_;
		PyObject* s2_;

		get_item_function get1_{nullptr};
		get_item_function get2_{nullptr};

		size_t cursor_;
		point point_;
	};

	template <typename Tp>
	mcore::linalg::vec
	build_l2_approximation(PyObject* container, size_t degree) {
		if (Py_ssize_t size = (Tp::get_size)(container)) {
			if (size == 1) {
				// insufficient number of samples;
				// let's generate an exception
				PyErr_SetString(PyExc_ValueError,
					"Number of samples must be greater then 1");
				throw wrapper_exception();
			}

			if (static_cast<size_t>(size) < degree) {
				// number of samples must be greater then
				// desired degree of a plynomial approxmation
				std::ostringstream oss;
				oss << "Insuffient number of samples (";
				oss << size;
				oss << "); degree of a polynom equals to " << degree;
				PyErr_SetString(PyExc_ValueError, oss.str().c_str());
				throw wrapper_exception();
			}

			pyobject_iterator_1<Tp::get_item> begin(container);
			pyobject_iterator_1<Tp::get_item> end(container, size);

			return np5::detail::approximate_l2(begin, end, degree);
		} else {
			PyErr_SetString(PyExc_ValueError,
				"Number of samples must be greater than 1");
			throw wrapper_exception();
		}
	}

	struct tuple_traits {
		static constexpr get_size_function get_size = PyTuple_Size;
		static constexpr get_item_function get_item = PyTuple_GetItem;
	};

	struct list_traits {
		static constexpr get_size_function get_size = PyList_Size;
		static constexpr get_item_function get_item = PyList_GetItem;
	};

	PyObject* build_result(mcore::linalg::vec&& v) noexcept {
		if (PyObject* result = PyTuple_New(v.dim())) {
			for (size_t i = 0; i < v.dim(); ++i) {
				if (PyObject* value = PyFloat_FromDouble(v(i))) {
					PyTuple_SetItem(result, i, value);
				} else {
					Py_DECREF(result);
					return nullptr;
				}
			}
			return result;
		} else
			return nullptr;
	}

	std::tuple<get_item_function, size_t> parse_argument(PyObject* obj) {
		get_item_function getter = nullptr;
		size_t size = 0;
		if (PyTuple_CheckExact(obj)) {
			size = PyTuple_Size(obj);
			getter = PyTuple_GetItem;
		} else if (PyList_CheckExact(obj)) {
			size = PyList_Size(obj);
			getter = PyList_GetItem;
		} else {
			PyErr_SetString(PyExc_ValueError,
				"Unsupported type of an argument");
			throw wrapper_exception();
		}
		return std::make_tuple(getter, size);
	}

} // anonymous namespace


static PyObject*
approximate_l2_bind(PyObject* self, PyObject* args) {
	PyObject* arg1 = nullptr;
	PyObject* arg2 = nullptr;
	size_t degree = 0;

	try {
		mcore::linalg::vec V;
		if (PyArg_ParseTuple(args, "Ol", &arg1, &degree)) {
			if (PyTuple_CheckExact(arg1))
				V = std::move(build_l2_approximation<tuple_traits>(arg1, degree));
			else if (PyList_CheckExact(arg1))
				V = std::move(build_l2_approximation<list_traits>(arg1, degree));
			else {
				// Unknown type of arguments
				PyErr_SetString(PyExc_ValueError,
					"Unsupported type of arguments");
				return nullptr;
			}
		} else if (PyArg_ParseTuple(args, "OOl", &arg1, &arg2, &degree)) {
			assert(arg1 != nullptr);
			assert(arg2 != nullptr);

			auto par1 = parse_argument(arg1);
			auto par2 = parse_argument(arg2);

			if (std::get<0>(par1) == std::get<0>(par2)) {
				size_t const size = std::get<1>(par1);
				if (size) {
					if (size > 1) {
						if (size > degree) {
							// both sizes are equal, so we can continue
							pyobject_iterator_2 begin(
								arg1, std::get<0>(par1),
								arg2, std::get<0>(par2));
							pyobject_iterator_2 end(arg1, arg2, size);
							V = np5::detail::approximate_l2(begin, end, degree);
						} else {
							// Wrong combination of the degree and the data
							PyErr_SetString(PyExc_ValueError,
								"Wrong combination of the degree and the data");
							return nullptr;
						}
					} else {
						// Unknown type of arguments
						PyErr_SetString(PyExc_ValueError,
							"Number of samples in both sequences are equal to 1");
						return nullptr;
					}
				} else {
					// Unknown type of arguments
					PyErr_SetString(PyExc_ValueError,
						"Number of samples in both sequences are equal to zero");
					return nullptr;
				}
			}

		} else {
			// Unknown type of arguments
			PyErr_SetString(PyExc_ValueError,
				"Unsupported type of arguments");
			return nullptr;
		}
		return build_result(std::move(V));
	}

	catch (...) {
		return nullptr;
	}
}


static PyObject*
approximate_l1_bind(PyObject* self, PyObject* args) {
	PyObject* arg1 = nullptr;
	PyObject* arg2 = nullptr;
	size_t degree = 0;

	//get_size_function get_size = nullptr;
	try {
		std::vector<point> pts;
		size_t size = 0;

		if (PyArg_ParseTuple(args, "Ol", &arg1, &degree)) {
			//if (PyTuple_CheckExact(arg1)) {
			//	get_size = PyTuple_Size;
			//	get_item = PyTuple_GetItem;
			//} else if (PyList_CheckExact(arg1)) {
			//	get_size = PyList_Size;
			//	get_item = PyList_GetItem;
			//} else {
			//	// Unknown type of arguments
			//	PyErr_SetString(PyExc_ValueError,
			//		"Unsupported type of arguments");
			//	return nullptr;
			//}
			auto par = parse_argument(arg1);

			size = std::get<1>(par);
			if (size) {
				if (size > 1) {
					if (size > degree) {
						pts.reserve(size);
						get_item_function get_item = std::get<0>(par);
						for (size_t i = 0; i < size; ++i)
							pts.emplace_back(point{get_item(arg1, i)});
					} else {
						PyErr_SetString(PyExc_ValueError,
							"Wrong combination of samples number and a degree");
						return nullptr;
					}
				} else {
					PyErr_SetString(PyExc_ValueError,
						"Data containes only one sample");
					return nullptr;
				}
			} else {
				PyErr_SetString(PyExc_ValueError,
					"Data is empty");
				return nullptr;
			}
		} else if (PyArg_ParseTuple(args, "OOl", &arg1, &arg2, &degree)) {
			auto par1 = parse_argument(arg1);
			auto par2 = parse_argument(arg2);

			if (std::get<1>(par1) != std::get<1>(par2)) {
				size_t const size = std::get<1>(par1);
				if (size) {
					if (size > 1 && degree < size) {
						pts.reserve(size);
						auto get1 = std::get<0>(par1);
						auto get2 = std::get<0>(par2);
						for (size_t i = 0; i < size; ++i)
							pts.emplace_back(point{get1(arg1, i), get2(arg2, i)});
					} else {
						PyErr_SetString(PyExc_ValueError,
							"Wrong combination of the degree and the data.");
						return nullptr;
					}
				} else {
					PyErr_SetString(PyExc_ValueError,
						"Empty data; nothing to approximate");
				}
			} else {
				PyErr_SetString(PyExc_ValueError,
					"Sequences are not aligned");
				return nullptr;
			}
		} else {
			PyErr_SetString(PyExc_ValueError,
				"Empty data; nothing to approximate");
			return nullptr;
		}

		return build_result(np5::detail::approximate_l1(
			std::begin(pts), std::end(pts), degree));
	}

	catch (...) {
		return nullptr;
	}
}


static PyMethodDef SpamMethods[] = {
	{"approximate_l2", approximate_l2_bind, METH_VARARGS,
		"Creates L2 (LSQ) data apprixmation."},
	{"approximate_l1", approximate_l1_bind, METH_VARARGS,
		"Calculates L1 data approximation"},
	{nullptr, nullptr, 0, nullptr}
};

PyMODINIT_FUNC
initregression(void) {
	(void) Py_InitModule("regression", SpamMethods);
}
