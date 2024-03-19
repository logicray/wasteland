#include <Python.h>
#include "example.c"

static PyObject* helloworld(PyObject* self) {
   return Py_BuildValue("s", "Hello, Python extensions!!");
}

static char helloworld_docs[] =
   "helloworld( ): Any message you want to put here!!\n";

static PyMethodDef helloworld_funcs[] = {
   {"helloworld", (PyCFunction)helloworld, METH_NOARGS, helloworld_docs},
   {"mul", (PyCFunction)example_mul, METH_VARARGS, mul_docs},
   {"div", (PyCFunction)example_div, METH_VARARGS, div_docs}, 
   {NULL},

};


static struct PyModuleDef helloworldModule =
{
    PyModuleDef_HEAD_INIT,
    "helloworld", /* name of module */
    helloworld_docs, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    helloworld_funcs
};

PyMODINIT_FUNC PyInit_helloworld(void)
{
    return PyModule_Create(&helloworldModule);
}
