# include <Python.h>

# include <cassert>
# include <cstddef>
# include <cstring>
# include <cstdarg>
# include <sstream>
# include <vector>

# include "polynomial.hpp"


namespace {

	class wrapper_exception {};

	struct point {
		point() {}

		point(PyObject* object) noexcept
			: x(PyFloat_AsDouble(PyTuple_GetItem(object, 0))),
			  y(PyFloat_AsDouble(PyTuple_GetItem(object, 1))) {}

		point& operator=(PyObject* object) {
			if (object) {
				x = PyFloat_AsDouble(PyTuple_GetItem(object, 0));
				y = PyFloat_AsDouble(PyTuple_GetItem(object, 1));
			}

			return *this;
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

	template <typename Tp>
	mcore::linalg::vec build_l2_approximation(PyObject* container, size_t degree) {
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


	typedef Py_ssize_t (*get_size_function)(PyObject*);
	typedef PyObject* (*get_item_function)(PyObject*, Py_ssize_t);

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

} // anonymous namespace


static PyObject*
approximate_l2_bind(PyObject* self, PyObject* args) {
	PyObject* arguments = nullptr;
	size_t degree = 0;

	try {
		mcore::linalg::vec V;
		if (PyArg_ParseTuple(args, "Ol", &arguments, &degree)) {
			if (PyTuple_CheckExact(arguments))
				V = std::move(build_l2_approximation<tuple_traits>(arguments, degree));
			else if (PyList_CheckExact(arguments))
				V = std::move(build_l2_approximation<list_traits>(arguments, degree));
			else {
				// Unknown type of arguments
				PyErr_SetString(PyExc_ValueError,
					"Unsupported type of arguments");
				return nullptr;
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
	PyObject* ps = nullptr;
	size_t degree = 0;

	get_size_function get_size = nullptr;
	get_item_function get_item = nullptr;
	try {
		if (PyArg_ParseTuple(args, "Ol", &ps, &degree)) {
			if (PyTuple_CheckExact(ps)) {
				get_size = PyTuple_Size;
				get_item = PyTuple_GetItem;
			} else if (PyList_CheckExact(ps)) {
				get_size = PyList_Size;
				get_item = PyList_GetItem;
			} else {
				// Unknown type of arguments
				PyErr_SetString(PyExc_ValueError,
					"Unsupported type of arguments");
				return nullptr;
			}
		}

		size_t size = get_size(ps);
		if (size) {
			if (size > 1) {
				if (size > degree) {
					std::vector<point> pts;
					pts.reserve(size);

					for (size_t i = 0; i < size; ++i)
						pts.emplace_back(point{get_item(ps, i)});

					return build_result(np5::detail::approximate_l1(
						std::begin(pts), std::end(pts), degree));

				} else {
					PyErr_SetString(PyExc_ValueError,
						"Wrong combination of samples number and a degree");
					return nullptr;
				}
			} else {
				PyErr_SetString(PyExc_ValueError,
					"Empty data; nothing to approximate");
				return nullptr;
			}
		} else {
			PyErr_SetString(PyExc_ValueError,
				"Empty data; nothing to approximate");
			return nullptr;
		}
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
