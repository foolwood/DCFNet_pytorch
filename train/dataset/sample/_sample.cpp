#include "Python.h"
#include "numpy/arrayobject.h"
#include "_sample.hpp"
#include "timer.hpp"
#include <omp.h>

static shape getShapeOfArray(PyArrayObject* obj) {
    shape s;
    s.ndim = PyArray_NDIM(obj);
    s.dims = PyArray_DIMS(obj);
    return s;
}

void print(shape s) {
    static char buffer[256];
    int shift = sprintf(buffer, "%d:[", (int)s.ndim);
    for (int i = 0 ; i < s.ndim; i++) {
        shift += sprintf(buffer+shift, "%d,", (int)s.dims[i]);
    }
    sprintf(buffer+shift, "]");
    puts(buffer);
}

void print_array(PyArrayObject *obj) {
    shape s = getShapeOfArray(obj);

    unsigned char* d = (unsigned char*)PyArray_GETPTR3(obj, 0, 0, 0);

    for (int i = 0 ; i < s.dims[0]; ++i) {
        for (int j = 0 ; j < s.dims[1]; ++j) {
            for (int k = 0 ; k < s.dims[2]; ++k) {
                unsigned char* data = (unsigned char*)PyArray_GETPTR3(obj, i, j, k);
                printf("%d %d %d %d %d\n", i, j, k, (int)(*data), (int)(*d));
                d++;
            }
        }
    }
}

// image: image data ptr
// s : h, w, c

inline uint8 saturate_cast(int v) {
    v = v > 0 ? v : 0;
    v = v > 255 ?  255 : v;
    return v;
}

extern void resize(uint8* image, shape s, uint8* data, int width, int height, Rect box, pixel mean);


static PyObject*
resample (PyObject *dummy, PyObject *args)
{
    PyObject *arg1 = NULL, *arg2, *arg3;
    PyObject *out = NULL;
    PyArrayObject *arr1 = NULL, *arr2 = NULL, *rect = NULL, *means = NULL;


    if (!PyArg_ParseTuple(args, "OOOO!", &arg1, &arg2, &arg3, &PyArray_Type, &out))
        return NULL;

    arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_UBYTE, NPY_IN_ARRAY);


    if (arr1 == NULL) {
        Py_XDECREF(arr1);
        return NULL;
    }


    arr2 = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_UBYTE, NPY_INOUT_ARRAY);

    if (arr2 == NULL) {
        Py_XDECREF(arr1);
        Py_XDECREF(arr2);
        return NULL;
    }

    rect = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_FLOAT32, NPY_IN_ARRAY);

    if (rect == NULL) {
        Py_XDECREF(arr1);
        Py_XDECREF(arr2);
        Py_XDECREF(rect);
        return NULL;
    }

    Rect r = get_rect(rect);

    means = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_UBYTE, NPY_IN_ARRAY);

    if (means == NULL) {
        Py_XDECREF(arr1);
        Py_XDECREF(arr2);
        Py_XDECREF(rect);
        Py_XDECREF(means);
        return NULL;
    } 

    pixel mean = get_mean(means);

    uint8 *image = (uint8*)PyArray_GETPTR3(arr1, 0, 0, 0);
    uint8 *data =  (uint8*)PyArray_GETPTR3(arr2, 0, 0, 0);

    // printf("ptr %p %p\n", data, (uint8*)PyArray_GETPTR3(arr2, 0, 0, 1));

    shape s1 = getShapeOfArray(arr1);
    shape s2 = getShapeOfArray(arr2);

    TS(resize);
    resize(image, s1, data, s2.dims[2], s2.dims[1], r, mean);
    TE(resize);

 
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(rect);
    Py_DECREF(means);
    Py_INCREF(Py_None);
    return Py_None;
}

Rect get_rect(PyArrayObject* obj) {
    Rect r;
    float* data = (float*)PyArray_GETPTR1(obj, 0);
    r.x1 = data[0];
    r.y1 = data[1];
    r.x2 = data[2];
    r.y2 = data[3];
    return r;
}

pixelf get_meanf(PyArrayObject* obj) {
    float* data = (float*)PyArray_GETPTR1(obj, 0);
    return (pixelf){data[0], data[1], data[2]};
}


pixel get_mean(PyArrayObject* obj) {
    uint8* data = (uint8*)PyArray_GETPTR1(obj, 0);
    return (pixel){data[0], data[1], data[2]};
}

static struct PyMethodDef methods[] = {
    {"resample", resample, METH_VARARGS, "sample func"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_sample (void)
{
    (void)Py_InitModule("_sample", methods);
    import_array();
}
