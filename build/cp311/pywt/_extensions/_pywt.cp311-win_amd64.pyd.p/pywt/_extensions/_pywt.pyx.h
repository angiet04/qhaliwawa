/* Generated by Cython 3.0.10 */

#ifndef __PYX_HAVE__pywt___extensions___pywt
#define __PYX_HAVE__pywt___extensions___pywt

#include "Python.h"
struct WaveletObject;
struct ContinuousWaveletObject;

/* "pywt/_extensions/_pywt.pxd":26
 *     have_c99_complex = 0
 * 
 * cdef public class Wavelet [type WaveletType, object WaveletObject]:             # <<<<<<<<<<<<<<
 *     cdef wavelet.DiscreteWavelet* w
 * 
 */
struct WaveletObject {
  PyObject_HEAD
  DiscreteWavelet *w;
  PyObject *name;
  PyObject *number;
};

/* "pywt/_extensions/_pywt.pxd":32
 *     cdef readonly number
 * 
 * cdef public class ContinuousWavelet [type ContinuousWaveletType, object ContinuousWaveletObject]:             # <<<<<<<<<<<<<<
 *     cdef wavelet.ContinuousWavelet* w
 * 
 */
struct ContinuousWaveletObject {
  PyObject_HEAD
  ContinuousWavelet *w;
  PyObject *name;
  PyObject *number;
  PyObject *dt;
};

#ifndef __PYX_HAVE_API__pywt___extensions___pywt

#ifdef CYTHON_EXTERN_C
    #undef __PYX_EXTERN_C
    #define __PYX_EXTERN_C CYTHON_EXTERN_C
#elif defined(__PYX_EXTERN_C)
    #ifdef _MSC_VER
    #pragma message ("Please do not define the '__PYX_EXTERN_C' macro externally. Use 'CYTHON_EXTERN_C' instead.")
    #else
    #warning Please do not define the '__PYX_EXTERN_C' macro externally. Use 'CYTHON_EXTERN_C' instead.
    #endif
#else
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C DL_IMPORT(PyTypeObject) WaveletType;
__PYX_EXTERN_C DL_IMPORT(PyTypeObject) ContinuousWaveletType;

#endif /* !__PYX_HAVE_API__pywt___extensions___pywt */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_pywt(void);
#else
/* WARNING: Use PyImport_AppendInittab("_pywt", PyInit__pywt) instead of calling PyInit__pywt directly from Python 3.5 */
PyMODINIT_FUNC PyInit__pywt(void);

#if PY_VERSION_HEX >= 0x03050000 && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || (defined(__cplusplus) && __cplusplus >= 201402L))
#if defined(__cplusplus) && __cplusplus >= 201402L
[[deprecated("Use PyImport_AppendInittab(\"_pywt\", PyInit__pywt) instead of calling PyInit__pywt directly.")]] inline
#elif defined(__GNUC__) || defined(__clang__)
__attribute__ ((__deprecated__("Use PyImport_AppendInittab(\"_pywt\", PyInit__pywt) instead of calling PyInit__pywt directly."), __unused__)) __inline__
#elif defined(_MSC_VER)
__declspec(deprecated("Use PyImport_AppendInittab(\"_pywt\", PyInit__pywt) instead of calling PyInit__pywt directly.")) __inline
#endif
static PyObject* __PYX_WARN_IF_PyInit__pywt_INIT_CALLED(PyObject* res) {
  return res;
}
#define PyInit__pywt() __PYX_WARN_IF_PyInit__pywt_INIT_CALLED(PyInit__pywt())
#endif
#endif

#endif /* !__PYX_HAVE__pywt___extensions___pywt */