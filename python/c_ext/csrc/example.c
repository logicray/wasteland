#include <Python.h>

static PyObject* example_mul(PyObject* self, PyObject* args)
{
	float a, b;
	if(!PyArg_ParseTuple(args, "ff", &a, &b))
	{
		return NULL;
	}
	return Py_BuildValue("f", a*b);
}


static PyObject* example_div(PyObject* self, PyObject* args)
{
	float a, b;
	if(!PyArg_ParseTuple(args, "ff", &a, &b))
	{
		return NULL;
	}
	return Py_BuildValue("f", a / b);
}

static char mul_docs[] = "mul(a, b): return a*b\n";
static char div_docs[] = "div(a, b): return a/b\n";

static PyMethodDef example_methods[] = 
{
	{"mul", (PyCFunction)example_mul, METH_VARARGS, mul_docs},
	{"div", (PyCFunction)example_div, METH_VARARGS, div_docs},
	{NULL, NULL, 0, NULL}
};


/*
static struct PyModuleDef helloworld =
{
    PyModuleDef_HEAD_INIT,
    "Combinations", // name of module 
    "usage: Combinations.uniqueCombinations(lstSortableItems, comboSize)\n", // module documentation, may be NULL 
    -1,    size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    example_methods
};

PyMODINIT_FUNC PyInit_helloworld(void)
{
    return PyModule_Create(&helloworld);
}
*/
