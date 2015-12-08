# include <Python.h>

# include <cassert>
# include <cstddef>
# include <cstring>
# include <cstdarg>

# include "polynomial.hpp"

# include <iostream>


namespace {

	template <PyObject* (*get_item)(PyObject*, Py_ssize_t)>
	class pyobject_iterator_1 {
		struct point {
			double x;
			double y;
		};

	public:
		pyobject_iterator_1(PyObject* object) noexcept
			: container_(object)
		{
			assert(container_ != nullptr);
			update_point();
		}

		explicit pyobject_iterator_1(PyObject* object, size_t size) noexcept
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
			update_point();
			return *this;
		}

		point const* operator->() const { return &point_; }

	private:
		void update_point() {
			if (PyObject* object = get_item(container_, cursor_)) {
				point_.x = PyFloat_AsDouble(PyTuple_GetItem(object, 0));
				point_.y = PyFloat_AsDouble(PyTuple_GetItem(object, 1));
			}
		}

	private:
		PyObject* container_;
		size_t cursor_{0};
		point point_;
	};

	template <typename Tp>
	mcore::linalg::vec build_l2_approximation(PyObject* container, size_t degree) {
		if (Py_ssize_t size = (Tp::get_size)(container)) {
			pyobject_iterator_1<Tp::get_item> begin(container);
			pyobject_iterator_1<Tp::get_item> end(container, size);

			return np5::detail::approximate_l2(begin, end, degree);
		} else
			return mcore::linalg::vec();
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

} // anonymous namespace

static PyObject*
approximate_l2_bind(PyObject* self, PyObject* args) {
	PyObject* arguments = nullptr;
	size_t degree = 0;

	mcore::linalg::vec V;
	if (PyArg_ParseTuple(args, "Ol", &arguments, &degree)) {
		if (PyTuple_CheckExact(arguments))
			V = std::move(build_l2_approximation<tuple_traits>(arguments, degree));
		else if (PyList_CheckExact(arguments))
			V = std::move(build_l2_approximation<list_traits>(arguments, degree));
	} else
		return nullptr;

	if (PyObject* result = PyTuple_New(degree + 1))  {
		for (size_t i = 0; i < degree + 1; ++i) {
			if (PyObject* value = PyFloat_FromDouble(V(i))) {
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


static PyMethodDef SpamMethods[] = {
	{"approximate_l2", approximate_l2_bind, METH_VARARGS, "Creates L2 (LSQ) data apprixmation."},
	{nullptr, nullptr, 0, nullptr}
};

PyMODINIT_FUNC
initregression(void) {
	(void) Py_InitModule("regression", SpamMethods);
}
