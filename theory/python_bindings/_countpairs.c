/* File: _countpairs.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Now, include the numpy header*/
#include <arrayobject.h>

//for correlation functions
#include "countpairs.h"
#include "countpairs_rp_pi.h"
#include "countpairs_wp.h"
#include "countpairs_xi.h"

//for the vpf
#include "countspheres.h"

//for the instruction set detection
#include "cpu_features.h"

//for unicode characters
#include "macros.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
//python3 follows
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#define INITERROR return NULL
PyObject *PyInit__countpairs(void);

#else
//python2 follows
#define GETSTATE(m) (&_state)
static struct module_state _state;
#define INITERROR return
PyMODINIT_FUNC init_countpairs(void);

#endif

#define NOTYPE_DESCR     (PyArray_DescrFromType(NPY_NOTYPE))

//File-scope variable
static int highest_isa;

//Docstrings for the methods
static char module_docstring[]             =    "Python extensions for calculating clustering statistics on simulations.\n"
    "\n"
    "countpairs       : Calculate the 3-D xi auto/cross-correlation function given two sets of arrays with Cartesian XYZ positions.\n"
    "countpairs_rp_pi : Calculate the 2-D DD("RP_CHAR","PI_CHAR") auto/cross-correlation function given two sets of arrays with Cartesian XYZ positions.\n"
    "countpairs_wp    : Calculate the projected auto-correlation function wp (assumes PERIODIC) given one set of arrays with Cartesian XYZ positions\n"
    "countpairs_xi    : Calculate the 3-d auto-correlation function xi (assumes PERIODIC) given one set of arrays with Cartesian XYZ positions\n"
    "countpairs_vpf   : Calculate the counts-in-spheres given one set of arrays with Cartesian XYZ positions\n"
    "\n"
    "See `Corrfunc/call_correlation_functions.py` for example calls to each function in the extension.\n";

/* static char error_out_docstring[]          =  "Error-handler for the module."; */

/* function proto-type*/
static PyObject *countpairs_countpairs(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_rp_pi(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_wp(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countpairs_xi(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_countspheres_vpf(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *countpairs_error_out(PyObject *module, const char *msg);

/* Inline documentation for the methods so that help(function) has something reasonably useful*/
static PyMethodDef module_methods[] = {
    /* {"countpairs_error_out"  ,(PyCFunction) countpairs_error_out        ,METH_VARARGS, error_out_docstring}, */
    {"countpairs"            ,(PyCFunction) countpairs_countpairs       ,METH_VARARGS | METH_KEYWORDS,
     "countpairs(autocorr, nthreads, binfile, X1, Y1, Z1, periodic=True,\n"
     "           X2=None, Y2=None, Z2=None, verbose=False, boxsize=0.0,\n"
     "           output_ravg=False, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculate the 3-D pair-counts, "XI_CHAR"(r), auto/cross-correlation \n"
     "function given two sets of points represented by X1/Y1/Z1 and X2/Y2/Z2 \n"
     "arrays.\n\n"
     
     "Note, that this module only returns pair counts and not the actual \n"
     "correlation function "XI_CHAR"(r). See the mocks/wtheta/wtheta.c for \n"
     "computing "XI_CHAR"(r) from the output of DD(r). Also note that the \n"
     "python wrapper for this extension: `Corrfunc.theory.DD` is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters \n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n\n"
     
     "autocorr : boolean\n"
     "   Boolean flag for auto/cross-correlation. If autocorr is set to 1,\n"
     "    are not used (but must still be passed, perhaps again as X1/Y1/Z1).\n"     
     "\n"
     "nthreads : integer\n"
     "   The number of OpenMP threads to use. Has no effect if OpenMP was not\n"
     "   enabled during library compilation.\n\n"

     "binfile : string\n"
     "   Filename specifying the ``r`` bins for ``DD``. The file should\n"
     "   contain white-space separated values  of (rmin, rmax)  for each\n"
     "   ``r`` wanted. The bins do not need to be contiguous but must be in\n"
     "   increasing order (smallest bins come first). \n\n"
     
     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "   The array of X/Y/Z positions for the first set of points.\n"
     "   Calculations are done in the precision of the supplied arrays.\n\n"

     "periodic : boolean\n"
     "   Boolean flag to indicate periodic boundary conditions.\n\n"

     "X2/Y2/Z2 : array-like, real (float/double)\n"
     "   Array of XYZ positions for the second set of points. *Must* be the same\n"
     "   precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.\n\n"

     "verbose : boolean (default false)\n"
     "   Boolean flag to control output of informational messages\n\n"

     "boxsize : double\n"
     "   The side-length of the cube in the cosmological simulation.\n"
     "   Present to facilitate exact calculations for periodic wrapping.\n"
     "   If boxsize is not supplied, then the wrapping is done based on\n"
     "   the maximum difference within each dimension of the X/Y/Z arrays.\n\n"
    
     "output_ravg : boolean (default false)\n"
     "   Boolean flag to output the average ``r`` for each bin. Code will\n"
     "   run slower if you set this flag. Also, note, if you are calculating\n"
     "   in single-precision, ``ravg`` will suffer from numerical loss of\n"
     "   precision and can not be trusted. If you need accurate ``ravg``\n"
     "   values, then pass in double precision arrays for the particle positions.\n\n"

     "c_api_timer : boolean (default false)\n"
     "   Boolean flag to measure actual time spent in the C libraries. Here\n"
     "   to allow for benchmarking and scaling studies.\n\n"

     "isa : integer (default -1)\n"
     "  Controls the runtime dispatch for the instruction set to use. Possible\n"
     "  options are: [-1, AVX, SSE42, FALLBACK]\n\n"
     "  Setting isa to -1 will pick the fastest available instruction\n"
     "  set on the current computer. However, if you set ``isa`` to, say,\n"
     "  ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "  revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n\n"
       
     "  Unless you are benchmarking the different instruction sets, you should\n"
     "  always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "  then the integer values correspond to the ``enum`` for the instruction set\n"
     "  defined in ``utils/defs.h``.\n\n"
       
    "Returns\n"
    "--------\n\n"
    "A tuple (results, time) \n\n"
     
    "results : A python list\n"
    "   A python list containing [rmin, rmax, ravg, npairs] for each radial bin\n"
    "   specified in the ``binfile``. If ``output_ravg`` is not set, then ``ravg``\n"
    "   will be set to 0.0 for all bins. ``npairs`` contains the number of pairs\n"
    "   in that bin and can be used to compute the actual "XI_CHAR"(r) by\n"
    "   combining with (DR, RR) counts.\n\n"

    "time : if ``c_api_timer`` is set, then the return value contains the time spent\n"
    "   in the API; otherwise time is set to 0.0\n\n"
     
    "Example\n"
    "-------\n\n"
     
    ">>> from Corrfunc._countpairs import countpairs\n"
    ">>> from Corrfunc.io import read_catalog\n" 
    ">>> x,y,z = read_catalog()\n"
    ">>> autocorr=1\n"
    ">>> nthreads=2\n"
    ">>> (DD, time) = countpairs(autocorr, nthreads, '../tests/bins',x, y, z, \n"
    "                            X2=x, Y2=y, Z2=z,verbose=True)\n"
    "\n"
    },
    {"countpairs_rp_pi"      ,(PyCFunction) countpairs_countpairs_rp_pi ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_rp_pi(autocorr, nthreads, pimax, binfile, X1, Y1, Z1,\n"
     "                 periodic=True, X2=None, Y2=None, Z2=None, verbose=False,\n"
     "                 boxsize=0.0, output_rpavg=False, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculate the 3-D pair-counts corresponding to the real-space correlation\n"
     "function, "XI_CHAR"("RP_CHAR", "PI_CHAR") or wp("RP_CHAR"). Pairs which are separated\n"
     "by less than the ``rp`` bins (specified in ``binfile``) in the X-Y plane, and\n"
     "less than ``pimax`` in the Z-dimension are counted.\n\n"

     "Note, that this module only returns pair counts and not the actual\n"
     "correlation function "XI_CHAR"("RP_CHAR", "PI_CHAR"). See ``theory/xi_rp_pi/wprp.c``\n"
     "for computing wp("RP_CHAR") from the pair counts returned by this module.\n"
     "Also note that the python wrapper for this extension: `Corrfunc.theory.DDrppi`\n"
     "is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n"
     "\n"
     "autocorr: boolean, required\n"
     "   Boolean flag for auto/cross-correlation. If autocorr is set to 1,\n"
     "    are not used (but must still be passed, perhaps again as X1/Y1/Z1).\n"     
     "\n"
     "nthreads: integer\n"
     "    The number of OpenMP threads to use. Has no effect if OpenMP was not\n"
     "    enabled during library compilation.\n"
     "\n"
     "pimax: double\n"
     "   A double-precision value for the maximum separation along\n"
     "   the Z-dimension. Note that only pairs with ``0 <= dz < pimax``\n"
     "   are counted (no equality).\n"
     "   Distances along the Z direction ("PI_CHAR") are binned with unit\n"
     "   depth. For instance, if ``pimax=40``, then 40 bins will be created\n"
     "   along the ``"PI_CHAR"`` direction.\n\n"

     "binfile : string\n"
     "   Filename specifying the ``rp`` bins for ``DDrppi``. The file should\n"
     "   contain white-space separated values  of (rpmin, rpmax)  for each\n"
     "   ``rp`` wanted. The bins do not need to be contiguous but must be in\n"
     "   increasing order (smallest bins come first). \n\n"
     
     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "   The array of X/Y/Z positions for the first set of points.\n"
     "   Calculations are done in the precision of the supplied arrays.\n"
     "\n"
     "periodic : boolean\n"
     "   Boolean flag to indicate periodic boundary conditions.\n"
     "\n"
     "X2/Y2/Z2 : array-like, real (float/double)\n"
     "   Array of XYZ positions for the second set of points. *Must* be the same\n"
     "   precision as the X1/Y1/Z1 arrays. Only required when ``autocorr==0``.\n"
     "\n"
     "verbose : boolean (default false)\n"
     "   Boolean flag to control output of informational messages\n"
     "\n"
     "boxsize : double\n"
     "   The side-length of the cube in the cosmological simulation.\n"
     "   Present to facilitate exact calculations for periodic wrapping.\n"
     "   If boxsize is not supplied, then the wrapping is done based on\n"
     "   the maximum difference within each dimension of the X/Y/Z arrays.\n"
     "\n"
     "output_rpavg : boolean (default false)\n"
     "   Boolean flag to output the average ``"RP_CHAR"`` for each bin. Code will\n"
     "   run slower if you set this flag. Also, note, if you are calculating\n"
     "   in single-precision, ``"RP_CHAR"`` will suffer from numerical loss of\n"
     "   precision and can not be trusted. If you need accurate ``"RP_CHAR"``\n"
     "   values, then pass in double precision arrays for the particle positions.\n"
     "\n"
     "c_api_timer : boolean (default false)\n"
     "   Boolean flag to measure actual time spent in the C libraries. Here\n"
     "   to allow for benchmarking and scaling studies.\n"
     "\n"
     "isa : integer (default -1)\n"
     "  Controls the runtime dispatch for the instruction set to use. Possible\n"
     "  options are: [-1, AVX, SSE42, FALLBACK]\n"
     "\n"
     "  Setting isa to -1 will pick the fastest available instruction\n"
     "  set on the current computer. However, if you set ``isa`` to, say,\n"
     "  ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "  revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "  Unless you are benchmarking the different instruction sets, you should\n"
     "  always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "  then the integer values correspond to the ``enum`` for the instruction set\n"
     "  defined in ``utils/defs.h``.\n"
     "\n"
     "Returns\n"
     "--------\n"
     "\n"
     "A tuple (results, time) \n"
     "\n"
     "results : A python list\n"
     "   A python list containing [rpmin, rpmax, rpavg, pimax, npairs] for each radial\n"
     "   bin specified in the ``binfile``. If ``output_rpavg`` is not set, then ``rpavg``\n"
     "   will be set to 0.0 for all bins. ``npairs`` contains the number of pairs\n"
     "   in that bin and can be used to compute the actual wp("RP_CHAR") by\n"
     "   combining with (DR, RR) counts.\n"
     "\n" 
     "time : if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "   in the API; otherwise time is set to 0.0\n"
     "\n"
     "Example\n"
     "--------\n"
     "\n"
     ">>> from Corrfunc._countpairs import countpairs_rp_pi\n"
     ">>> from Corrfunc.io import read_catalog\n" 
     ">>> x,y,z = read_catalog()\n"
     ">>> autocorr=1\n"
     ">>> nthreads=2\n"
     ">>> pimax=40.0\n"
     ">>> (DDrppi, time) = countpairs_rp_pi(autocorr, nthreads, pimax, '../tests/bins',\n"
     "                                      x, y, z, X2=x, Y2=y, Z2=z,\n"
     "                                      verbose=True, output_rpavg=True)\n"
     "\n"
    },
    {"countpairs_wp"         ,(PyCFunction) countpairs_countpairs_wp    ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_wp(boxsize, pimax, nthreads, binfile, X, Y, Z, verbose=False,\n"
     "              output_rpavg=False, c_api_timer=False, isa=-1)\n"
     "\n"
     "Function to compute the projected correlation function in a periodic\n"
     "cosmological box. Pairs which are separated by less than the ``"RP_CHAR"``\n"
     "bins (specified in ``binfile``) in the X-Y plane, and less than ``"PIMAX_CHAR"``\n"
     "in the Z-dimension are counted. *Always* uses ``PERIODIC`` boundary conditions.\n"
     "\n"
     "This module returns the actual correlation function using the natural estimator.\n"
     "Analytic randoms are used to compute wp("RP_CHAR") from the pair counts. If you\n"
     "need a different estimator, Landy-Szalay, for instance, then you should compute\n"
     "the raw pair counts with the module ``countpairs_rp_pi`` and then calculate the\n"
     "Landy-Szalay estimator for wp("RP_CHAR").\n"
     "\n"
     "Note that pairs are double-counted. And if ``rpmin`` is set to\n"
     "0.0, then all the self-pairs (i'th particle with itself) are\n"
     "added to the first bin => minimum number of pairs in the first bin\n"
     "is the total number of particles. Also note that the python wrapper\n"
     "for this extension: `Corrfunc.theory.wp` is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n"
     "\n"
     "boxsize: double\n"
     "   A double-precision value for the boxsize of the simulation\n"
     "   in same units as the particle positions and the ``rp`` bins.\n"
     "\n"
     "pimax: double\n"
     "   A double-precision value for the maximum separation along\n"
     "   the Z-dimension. Note that only pairs with ``0 <= dz < pimax``\n"
     "   are counted (no equality).\n"
     "   Distances along the Z direction ("PI_CHAR") are binned with unit\n"
     "   depth. For instance, if ``pimax=40``, then 40 bins will be created\n"
     "   along the ``"PI_CHAR"`` direction.\n"
     "\n"
     "nthreads: integer\n"
     "   Number of threads to use.\n"
     "\n"
     "binfile : string\n"
     "   Filename specifying the ``rp`` bins for ``wp``. The file should\n"
     "   contain white-space separated values  of (rpmin, rpmax)  for each\n"
     "   ``rp`` wanted. The bins do not need to be contiguous but must be in\n"
     "   increasing order (smallest bins come first). \n"
     "\n"
     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "   The array of X/Y/Z positions for the first set of points.\n"
     "   Calculations are done in the precision of the supplied arrays.\n"
     "\n"
     "verbose : boolean (default false)\n"
     "   Boolean flag to control output of informational messages\n"
     "\n"
     "output_rpavg : boolean (default false)\n"
     "   Boolean flag to output the average ``"RP_CHAR"`` for each bin. Code will\n"
     "   run slower if you set this flag. Also, note, if you are calculating\n"
     "   in single-precision, ``"RP_CHAR"`` will suffer from numerical loss of\n"
     "   precision and can not be trusted. If you need accurate ``"RP_CHAR"``\n"
     "   values, then pass in double precision arrays for the particle positions.\n"
     "\n"
     "c_api_timer : boolean (default false)\n"
     "   Boolean flag to measure actual time spent in the C libraries. Here\n"
     "   to allow for benchmarking and scaling studies.\n"
     "\n"
     "isa : integer (default -1)\n"
     "  Controls the runtime dispatch for the instruction set to use. Possible\n"
     "  options are: [-1, AVX, SSE42, FALLBACK]\n"
     "\n"
     "  Setting isa to -1 will pick the fastest available instruction\n"
     "  set on the current computer. However, if you set ``isa`` to, say,\n"
     "  ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "  revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "  Unless you are benchmarking the different instruction sets, you should\n"
     "  always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "  then the integer values correspond to the ``enum`` for the instruction set\n"
     "  defined in ``utils/defs.h``.\n"
     "\n"
     "Returns\n"
     "--------\n"
     "\n"
     "A tuple (results, time) \n"
     "\n"
     "results : A python list\n"
     "   A python list containing [rpmin, rpmax, rpavg, wp, npairs] for each radial\n"
     "   bin specified in the ``binfile``. If ``output_rpavg`` is not set then\n"
     "   ``rpavg`` will be set to 0.0 for all bins. ``wp`` contains the projected\n"
     "   correlation function while ``npairs`` contains the number of unique pairs\n"
     "   in that bin.\n"
     "\n" 
     "time : if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "   in the API; otherwise time is set to 0.0\n"
     "\n"
     "Example\n"
     "--------\n"
     "\n"
     ">>> from _countpairs import countpairs_wp\n"
     ">>> from Corrfunc.io import read_catalog\n" 
     ">>> x,y,z = read_catalog()\n"
     ">>> nthreads=2\n"
     ">>> pimax=40.0\n"
     ">>> boxsize = 420.0\n"
     ">>> (wp, time) = countpairs_wp(boxsize, nthreads, pimax, '../tests/bins',\n"
     "                               x, y, z, verbose=True, output_rpavg=True)\n"
     "\n"
    },
    {"countpairs_xi"         ,(PyCFunction) countpairs_countpairs_xi    ,METH_VARARGS | METH_KEYWORDS,
     "countpairs_xi(boxsize, nthreads, binfile, X, Y, Z, verbose=False,\n"
     "              output_ravg=False, c_api_timer=False, isa=-1)\n"
     "\n"
     "Function to compute the projected correlation function in a periodic\n"
     "cosmological box. Pairs which are separated by less than the ``r``\n"
     "bins (specified in ``binfile``). *Always* uses ``PERIODIC`` boundary conditions.\n"
     "\n"
     "This module returns the actual correlation function using the natural estimator.\n"
     "Analytic randoms are used to compute "XI_CHAR"(r) from the pair counts. If you\n"
     "need a different estimator, Landy-Szalay, for instance, then you should compute\n"
     "the raw pair counts with the module ``countpairs`` and then calculate the\n"
     "Landy-Szalay estimator for "XI_CHAR"(r).\n"
     "\n"
     "Note that pairs are double-counted. And if ``rmin`` is set to\n"
     "0.0, then all the self-pairs (i'th particle with itself) are\n"
     "added to the first bin => minimum number of pairs in the first bin\n"
     "is the total number of particles. Also note that the python wrapper\n"
     "for this extension: `Corrfunc.theory.xi` is more user-friendly.\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n"
     "\n"
     "boxsize: double\n"
     "   A double-precision value for the boxsize of the simulation\n"
     "   in same units as the particle positions and the ``r`` bins.\n"
     "\n"
     "nthreads: integer\n"
     "   Number of threads to use.\n"
     "\n"
     "binfile : string\n"
     "   Filename specifying the ``r`` bins for ``xi``. The file should\n"
     "   contain white-space separated values  of (rmin, rmax)  for each\n"
     "   ``r`` wanted. The bins do not need to be contiguous but must be in\n"
     "   increasing order (smallest bins come first). \n"
     "\n"
     "X1/Y1/Z1 : array-like, real (float/double)\n"
     "   The array of X/Y/Z positions for the first set of points.\n"
     "   Calculations are done in the precision of the supplied arrays.\n"
     "\n"
     "verbose : boolean (default false)\n"
     "   Boolean flag to control output of informational messages\n"
     "\n"
     "output_ravg : boolean (default false)\n"
     "   Boolean flag to output the average ``r`` for each bin. Code will\n"
     "   run slower if you set this flag. Also, note, if you are calculating\n"
     "   in single-precision, ``r`` will suffer from numerical loss of\n"
     "   precision and can not be trusted. If you need accurate ``r``\n"
     "   values, then pass in double precision arrays for the particle positions.\n"
     "\n"
     "c_api_timer : boolean (default false)\n"
     "   Boolean flag to measure actual time spent in the C libraries. Here\n"
     "   to allow for benchmarking and scaling studies.\n"
     "\n"
     "isa : integer (default -1)\n"
     "  Controls the runtime dispatch for the instruction set to use. Possible\n"
     "  options are: [-1, AVX, SSE42, FALLBACK]\n"
     "\n"
     "  Setting isa to -1 will pick the fastest available instruction\n"
     "  set on the current computer. However, if you set ``isa`` to, say,\n"
     "  ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "  revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "  Unless you are benchmarking the different instruction sets, you should\n"
     "  always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "  then the integer values correspond to the ``enum`` for the instruction set\n"
     "  defined in ``utils/defs.h``.\n"
     "\n"
     "Returns\n"
     "--------\n"
     "\n"
     "A tuple (results, time) \n"
     "\n"
     "results : A python list\n"
     "   A python list containing [rmin, rmax, ravg, xi, npairs] for each radial\n"
     "   bin specified in the ``binfile``. If ``output_ravg`` is not set then\n"
     "   ``ravg`` will be set to 0.0 for all bins. ``xi`` contains the projected\n"
     "   correlation function while ``npairs`` contains the number of unique pairs\n"
     "   in that bin.\n"
     "\n" 
     "time : if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "   in the API; otherwise time is set to 0.0\n"
     "\n"
     "Example\n"
     "--------\n"
     "\n"
     ">>> from _countpairs import countpairs_xi\n"
     ">>> from Corrfunc.io import read_catalog\n" 
     ">>> x,y,z = read_catalog()\n"
     ">>> nthreads=2\n"
     ">>> boxsize = 420.0\n"
     ">>> (xi, time) = countpairs_xi(boxsize, nthreads, '../tests/bins',\n"
     "                               x, y, z, verbose=True, output_ravg=True)\n"
     "\n"
    },
    {"countspheres_vpf"      ,(PyCFunction) countpairs_countspheres_vpf ,METH_VARARGS | METH_KEYWORDS,
     "countspheres_vpf(rmax, nbins, nspheres, numpN, seed,\n"
     "                 X, Y, Z, verbose=False, periodic=True,\n"
     "                 boxsize=0.0, c_api_timer=False, isa=-1)\n"
     "\n"
     "Calculates the fraction of random spheres that contain exactly *N* points, pN(r).\n"
     "\n"
     UNICODE_WARNING
     "\n"
     "Parameters\n"
     "-----------\n"
     "Every parameter can be passed as a keyword of the corresponding name.\n"
     "\n"
     "rmax: double\n"
     "   Maximum radius of the sphere to place on the particles\n"
     "\n"
     "nbins: integer\n"
     "   Number of bins in the counts-in-cells. Radius of first shell\n"
     "    is rmax/nbins\n"
     "\n"
     "nspheres: integer (>= 0)\n"
     "   Number of random spheres to place within the particle distribution.\n"
     "   For a small number of spheres, the error is larger in the measured\n"
     "   pN's.\n"
     "\n"
     "numpN: integer (>= 1)\n"
     "   Governs how many unique pN's are to returned. If ``numpN` is set to 1,\n"
     "   then only the vpf (p0) is returned. For ``numpN=2``, p0 and p1 are\n"
     "   returned.\n"
     "\n"
     "   More explicitly, the columns in the results look like the following:\n"
     "     numpN = 1 -> p0\n"
     "     numpN = 2 -> p0 p1\n"
     "     numpN = 3 -> p0 p1 p2\n"
     "     and so on...(note that p0 is the vpf).\n"
     "\n"
     "seed: unsigned integer\n"
     "   Random number seed for the underlying GSL random number generator. Used\n"
     "   to draw centers of the spheres.\n"
     "\n"
     "X/Y/Z: arraytype, real (float/double)\n"
     "   Particle positions in the 3 axes. Must be within [0, boxsize]\n"
     "   and specified in the same units as ``rp_bins`` and boxsize. All\n"
     "   3 arrays must be of the same floating-point type.\n"
     "   \n"
     "   Calculations will be done in the same precision as these arrays,\n"
     "   i.e., calculations will be in floating point if XYZ are single\n"
     "   precision arrays (C float type); or in double-precision if XYZ\n"
     "   are double precision arrays (C double type).\n"
     "\n"
     "verbose: boolean (default false)\n"
     "   Boolean flag to control output of informational messages\n"
     "\n"
     "periodic: boolean\n"
     "    Boolean flag to indicate periodic boundary conditions.\n"
     "\n"
     "boxsize: double\n"
     "    The side-length of the cube in the cosmological simulation.\n"
     "    Present to facilitate exact calculations for periodic wrapping.\n"
     "    If boxsize is not supplied, then the wrapping is done based on\n"
     "    the maximum difference within each dimension of the X/Y/Z arrays.\n"
     "c_api_timer : boolean (default false)\n"
     "   Boolean flag to measure actual time spent in the C libraries. Here\n"
     "   to allow for benchmarking and scaling studies.\n"
     "\n"
     "isa : integer (default -1)\n"
     "  Controls the runtime dispatch for the instruction set to use. Possible\n"
     "  options are: [-1, AVX, SSE42, FALLBACK]\n"
     "\n"
     "  Setting isa to -1 will pick the fastest available instruction\n"
     "  set on the current computer. However, if you set ``isa`` to, say,\n"
     "  ``AVX`` and ``AVX`` is not available on the computer, then the code will\n"
     "  revert to using ``FALLBACK`` (even though ``SSE42`` might be available).\n"
     "\n"
     "  Unless you are benchmarking the different instruction sets, you should\n"
     "  always leave ``isa`` to the default value. And if you *are* benchmarking,\n"
     "  then the integer values correspond to the ``enum`` for the instruction set\n"
     "  defined in ``utils/defs.h``.\n"
     "\n"
     "Returns\n"
     "--------\n"
     "\n"
     "A tuple (results, time) \n"
     "\n"
     "results : a Python list \n"
     "   The list contains [rmax, p0, p1,..., p(num_pN-1)] for each radial bin.\n"
     "\n"
     "time : if ``c_api_timer`` is set, then the return value contains the time spent\n"
     "   in the API; otherwise time is set to 0.0\n"
     "\n"
     "Example\n"
     "--------\n"
     "\n"
     ">>> from _countpairs import countspheres_vpf\n"
     ">>> from Corrfunc.io import read_catalog\n"
     ">>> rmax = 10.0\n"
     ">>> nbins = 10\n"
     ">>> nspheres = 10000\n"
     ">>> numpN = 8\n"
     ">>> seed = -1\n"
     ">>> boxsize = 420.0\n"
     ">>> X, Y, Z = read_catalog()\n"
     ">>> results, api_time = countspheres_vpf(rmax, nbins, nspheres, numpN, seed,\n"
     "                                         X, Y, Z,\n"
     "                                         verbose=True,\n"
     "                                         c_api_timer=True,\n"
     "                                         boxsize=boxsize,\n"
     "                                         periodic=True)\n"
     "\n"
    },
    {NULL, NULL, 0, NULL}
};

static PyObject *countpairs_error_out(PyObject *module, const char *msg)
{
#if PY_MAJOR_VERSION < 3
    (void) module;//to avoid unused warning with python2
#endif    

    struct module_state *st = GETSTATE(module);
    PyErr_SetString(st->error, msg);
    PyErr_Print();
    Py_RETURN_NONE;
}


#if PY_MAJOR_VERSION >= 3
static int _countpairs_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _countpairs_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_countpairs",
    module_docstring,
    sizeof(struct module_state),
    module_methods,
    NULL,
    _countpairs_traverse,
    _countpairs_clear,
    NULL
};


PyObject *PyInit__countpairs(void)
#else
//Python 2
PyMODINIT_FUNC init_countpairs(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3("_countpairs", module_methods, module_docstring);
#endif

    if (module == NULL) {
        INITERROR;
    }

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_countpairs.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    
    /* Load `numpy` functionality. */
    import_array();

    highest_isa = instrset_detect();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif

}

    
static int64_t check_dims_and_datatype(PyObject *module, PyArrayObject *x1_obj, PyArrayObject *y1_obj, PyArrayObject *z1_obj, size_t *element_size)
{
    char msg[1024];

    /* All the arrays should be 1-D*/
    const int nxdims = PyArray_NDIM(x1_obj);
    const int nydims = PyArray_NDIM(y1_obj);
    const int nzdims = PyArray_NDIM(z1_obj);

    if(nxdims != 1 || nydims != 1 || nzdims != 1) {
        snprintf(msg, 1024, "ERROR: Expected 1-D numpy arrays.\nFound (nxdims, nydims, nzdims) = (%d, %d, %d) instead",
                 nxdims, nydims, nzdims);
        countpairs_error_out(module, msg);
        return -1;
    }

    /* All the arrays should be floating point (only float32 and float64 are allowed) */
    const int x_type = PyArray_TYPE(x1_obj);
    const int y_type = PyArray_TYPE(y1_obj);
    const int z_type = PyArray_TYPE(z1_obj);
    if( ! ((x_type == NPY_FLOAT || x_type == NPY_DOUBLE) &&
           (y_type == NPY_FLOAT || y_type == NPY_DOUBLE) &&
           (z_type == NPY_FLOAT || z_type == NPY_DOUBLE))
        ) {
        PyArray_Descr *x_descr = PyArray_DescrFromType(x_type);
        PyArray_Descr *y_descr = PyArray_DescrFromType(y_type);
        PyArray_Descr *z_descr = PyArray_DescrFromType(z_type);
        if(x_descr == NULL || y_descr == NULL || z_descr == NULL) {
            /* Generating the dtype descriptor failed somehow. At least provide some information */
            snprintf(msg, 1024, "TypeError: Expected 3 floating point arrays (allowed types = %d or %d). Instead found type-nums (%d, %d, %d)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type);
        } else {
            snprintf(msg, 1024, "TypeError: Expected 3 floating point arrays (allowed types = %d or %d). Instead found type-nums (%d, %d, %d) "
                     "with type-names = (%s, %s, %s)\n",
                     NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type, x_descr->typeobj->tp_name, y_descr->typeobj->tp_name, z_descr->typeobj->tp_name);
        }
        Py_XDECREF(x_descr);Py_XDECREF(y_descr);Py_XDECREF(z_descr);
        countpairs_error_out(module, msg);
        return -1;
    }

    if( x_type != y_type || y_type != z_type) {
        PyArray_Descr *x_descr = PyArray_DescrFromType(x_type);
        PyArray_Descr *y_descr = PyArray_DescrFromType(y_type);
        PyArray_Descr *z_descr = PyArray_DescrFromType(z_type);
        if(x_descr == NULL || y_descr == NULL || z_descr == NULL) {
          /* Generating the dtype descriptor failed somehow. At least provide some information */
          snprintf(msg, 1024, "TypeError: Expected *ALL* 3 floating point arrays to be the same type (allowed types = %d or %d). Instead found type-nums (%d, %d, %d)\n",
                   NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type);
        } else {
          snprintf(msg, 1024, "TypeError: Expected *ALL* 3 floating point arrays to be the same type (allowed types = %d or %d). Instead found type-nums (%d, %d, %d) "
                   "with type-names = (%s, %s, %s)\n",
                   NPY_FLOAT, NPY_DOUBLE, x_type, y_type, z_type, x_descr->typeobj->tp_name, y_descr->typeobj->tp_name, z_descr->typeobj->tp_name);
        }
        Py_XDECREF(x_descr);Py_XDECREF(y_descr);Py_XDECREF(z_descr);
        countpairs_error_out(module, msg);
        return -1;
    }
    
    /* Check if the number of elements in the 3 Python arrays are identical */
    const int64_t nx1 = (int64_t)PyArray_SIZE(x1_obj);
    const int64_t ny1 = (int64_t)PyArray_SIZE(y1_obj);
    const int64_t nz1 = (int64_t)PyArray_SIZE(z1_obj);

    if(nx1 != ny1 || ny1 != nz1) {
      snprintf(msg, 1024, "ERROR: Expected arrays to have the same number of elements in all 3-dimensions.\nFound (nx, ny, nz) = (%"PRId64", %"PRId64", %"PRId64") instead",
               nx1, ny1, nz1);
      countpairs_error_out(module, msg);
      return -1;
    }


    /* Return the size of each element of the data object */
    if(x_type == NPY_FLOAT) {
      *element_size = sizeof(float);
    } else {
      *element_size = sizeof(double);
    }
    
    return nx1;
}

static int print_kwlist_into_msg(char *msg, const size_t totsize, size_t len, char *kwlist[], const size_t nitems)
{
    for(size_t i=0;i<nitems;i++) {
        
        if(len+strlen(kwlist[i]) >= totsize-2) {
            return EXIT_FAILURE;
        }
        
        memcpy(msg+len, kwlist[i], strlen(kwlist[i]));
        len += strlen(kwlist[i]);
        msg[len] = ',';
        msg[len+1] = ' ';
        len += 2;
    }
    
    msg[len]='\0';
    return EXIT_SUCCESS;
}


static PyObject *countpairs_countpairs(PyObject *self, PyObject *args, PyObject *kwargs)
{
    //Error-handling is global in python2 -> stored in struct module_state _struct declared at the top of this file
#if PY_MAJOR_VERSION < 3
    (void) self;
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash. 
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif    
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *x2_obj=NULL, *y2_obj=NULL, *z2_obj=NULL;

    int autocorr=0;
    int nthreads=4;
    char *binfile;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.periodic = 1;
    options.need_avg_sep = 0;
    options.c_api_timer = 0;
    static char *kwlist[] = {
        "autocorr",
        "nthreads",
        "binfile",
        "X1",
        "Y1",
        "Z1",
        "X2",
        "Y2",
        "Z2",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",
        "output_ravg",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX, SSE, FALLBACK */
        NULL
    };


    if ( ! PyArg_ParseTupleAndKeywords(args, kwargs, "iisO!O!O!|O!O!O!bbdbbi", kwlist,
                                       &autocorr,&nthreads,&binfile,
                                       &PyArray_Type,&x1_obj,
                                       &PyArray_Type,&y1_obj,
                                       &PyArray_Type,&z1_obj,
                                       &PyArray_Type,&x2_obj,
                                       &PyArray_Type,&y2_obj,
                                       &PyArray_Type,&z2_obj,
                                       &(options.periodic),
                                       &(options.verbose),
                                       &(options.boxsize),
                                       &(options.need_avg_sep),
                                       &(options.c_api_timer),
                                       &(options.instruction_set))

         ) {
        
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In DD> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }
        
        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }
    
    /* We have numpy arrays and all the required inputs*/
    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return 
        Py_RETURN_NONE;
    }

    int64_t ND2 = 0;
    if(autocorr == 0) {
        char msg[1024];
        if(x2_obj == NULL || y2_obj == NULL || z2_obj == NULL) {
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, need to pass the second set of positions (X2=numpy array, Y2=numpy array, Z2=numpy array).\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        size_t element_size2;
        ND2 = check_dims_and_datatype(module, x2_obj, y2_obj, z2_obj, &element_size2);
        if(ND2 == -1) {
            //Error has already been set -> simply return 
            Py_RETURN_NONE;
        }
        if(element_size != element_size2) {
            snprintf(msg, 1024, "TypeError: In %s: The two arrays must have the same data-type. First array is of type %s while second array is of type %s\n",
                     __FUNCTION__, element_size == 4 ? "floats":"doubles", element_size2 == 4 ? "floats":"doubles");
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
    }

    
    /* 
       Interpret the input objects as numpy arrays (of whatever the input type the python object has). 
       NULL initialization is necessary since we might be calling XDECREF.
       The input objects can be converted into the required DOUBLE array.
    */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = NULL, *y1_array = NULL, *z1_array = NULL;
    x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);
    
    /* NULL initialization is necessary since we might be calling XDECREF*/
    PyObject *x2_array = NULL, *y2_array = NULL, *z2_array = NULL;
    if(autocorr == 0) {
        x2_array = PyArray_FromArray(x2_obj, NOTYPE_DESCR, requirements);
        y2_array = PyArray_FromArray(y2_obj, NOTYPE_DESCR, requirements);
        z2_array = PyArray_FromArray(z2_obj, NOTYPE_DESCR, requirements);
    }
    if (x1_array == NULL || y1_array == NULL || z1_array == NULL ||
        (autocorr==0 && (x2_array == NULL || y2_array == NULL || z2_array == NULL))) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        
        Py_XDECREF(x2_array);
        Py_XDECREF(y2_array);
        Py_XDECREF(z2_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input to arrays of allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data */
    void *X1 = NULL, *Y1=NULL, *Z1=NULL;    
    X1 = PyArray_DATA((PyArrayObject *) x1_array); 
    Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    void *X2 = NULL, *Y2=NULL, *Z2=NULL;
    if(autocorr==0) {
        X2 = PyArray_DATA((PyArrayObject *) x2_array);
        Y2 = PyArray_DATA((PyArrayObject *) y2_array);
        Z2 = PyArray_DATA((PyArrayObject *) z2_array);
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    results_countpairs results;
    options.float_type = element_size;
    double c_api_time = 0.0;
    int status = countpairs(ND1,X1,Y1,Z1,
                            ND2,X2,Y2,Z2,
                            nthreads,
                            autocorr,
                            binfile,
                            &results,
                            &options, NULL);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    Py_XDECREF(x2_array);Py_XDECREF(y2_array);Py_XDECREF(z2_array);

    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        PyObject *item = NULL;
        const double rpavg = results.rpavg[i];
        item = Py_BuildValue("(dddk)", rlow,results.rupp[i],rpavg,results.npairs[i]);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }

    free_results(&results);
    return Py_BuildValue("(Od)", ret, c_api_time);
}


static PyObject *countpairs_countpairs_rp_pi(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash. 
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif    
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    PyArrayObject *x2_obj=NULL, *y2_obj=NULL, *z2_obj=NULL;
    int autocorr=0;
    int nthreads=4;
    
    double pimax;
    char *binfile;
    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.periodic = 1;
    options.c_api_timer = 0;
    static char *kwlist[] = {
        "autocorr",
        "nthreads",
        "pimax",
        "binfile",
        "X1",
        "Y1",
        "Z1",
        "X2",
        "Y2",
        "Z2",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",
        "output_rpavg",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX, SSE, FALLBACK */
        NULL
    };

    if ( ! PyArg_ParseTupleAndKeywords(args, kwargs, "iidsO!O!O!|O!O!O!bbdbbi", kwlist,
                                       &autocorr,&nthreads,&pimax,&binfile,
                                       &PyArray_Type,&x1_obj,
                                       &PyArray_Type,&y1_obj,
                                       &PyArray_Type,&z1_obj,
                                       &PyArray_Type,&x2_obj,
                                       &PyArray_Type,&y2_obj,
                                       &PyArray_Type,&z2_obj,
                                       &(options.periodic),
                                       &(options.verbose),
                                       &(options.boxsize),
                                       &(options.need_avg_sep),
                                       &(options.c_api_timer),
                                       &(options.instruction_set))

         ) {
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In DDrppi> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }
        
        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    options.autocorr=autocorr;
    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    size_t element_size;
    /* How many data points are there? And are they all of floating point type */
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return 
        Py_RETURN_NONE;
    }

    int64_t ND2=ND1;
    if(autocorr == 0) {
        char msg[1024];
        if(x2_obj == NULL || y2_obj == NULL || z2_obj == NULL) {
            snprintf(msg, 1024, "ValueError: In %s: If autocorr is 0, need to pass the second set of positions (X2=numpy array, Y2=numpy array, Z2=numpy array).\n",
                     __FUNCTION__);
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
        size_t element_size2;
        ND2 = check_dims_and_datatype(module, x2_obj, y2_obj, z2_obj, &element_size2);
        if(ND2 == -1) {
            //Error has already been set -> simply return 
            Py_RETURN_NONE;
        }
        if(element_size != element_size2) {
            snprintf(msg, 1024, "TypeError: In %s: The two arrays must have the same data-type. First array is of type %s while second array is of type %s\n",
                     __FUNCTION__, element_size == 4 ? "floats":"doubles", element_size2 == 4 ? "floats":"doubles");
            countpairs_error_out(module, msg);
            Py_RETURN_NONE;
        }
    } 
    
    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = NULL, *y1_array = NULL, *z1_array = NULL;
    PyObject *x2_array = NULL, *y2_array = NULL, *z2_array = NULL;
    x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if(autocorr == 0) {
        x2_array = PyArray_FromArray(x2_obj, NOTYPE_DESCR, requirements);
        y2_array = PyArray_FromArray(y2_obj, NOTYPE_DESCR, requirements);
        z2_array = PyArray_FromArray(z2_obj, NOTYPE_DESCR, requirements);
    }

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL ||
        (autocorr == 0 && (x2_array == NULL || y2_array == NULL || z2_array == NULL))) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);

        Py_XDECREF(x2_array);
        Py_XDECREF(y2_array);
        Py_XDECREF(z2_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input to arrays of allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data as C-types. */
    void *X1 = NULL, *Y1 = NULL, *Z1 = NULL;
    void *X2 = NULL, *Y2 = NULL, *Z2 = NULL;
    X1 = PyArray_DATA((PyArrayObject *) x1_array); 
    Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    if(autocorr == 0) {
        X2 = PyArray_DATA((PyArrayObject *) x2_array);
        Y2 = PyArray_DATA((PyArrayObject *) y2_array);
        Z2 = PyArray_DATA((PyArrayObject *) z2_array);
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;
    
    options.float_type = element_size;
    results_countpairs_rp_pi results;
    double c_api_time = 0.0;
    int status = countpairs_rp_pi(ND1,X1,Y1,Z1,
                                  ND2,X2,Y2,Z2,
                                  nthreads,
                                  autocorr,
                                  binfile,
                                  pimax,
                                  &results,
                                  &options, NULL);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;
    
    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);//x1 should absolutely not be NULL
    Py_XDECREF(x2_array);Py_XDECREF(y2_array);Py_XDECREF(z2_array);//x2 might be NULL depending on value of autocorr
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


    /* Build the output list */
    PyObject *ret = PyList_New(0);//create an empty list
    double rlow=results.rupp[0];
    const double dpi = pimax/(double)results.npibin ;

    for(int i=1;i<results.nbin;i++) {
        for(int j=0;j<results.npibin;j++) {
            const int bin_index = i*(results.npibin + 1) + j;
            PyObject *item = NULL;
            const double rpavg = results.rpavg[bin_index];
            item = Py_BuildValue("(ddddk)", rlow,results.rupp[i],rpavg,(j+1)*dpi,results.npairs[bin_index]);
            PyList_Append(ret, item);
            Py_XDECREF(item);
        }
        rlow=results.rupp[i];
    }
    free_results_rp_pi(&results);
    
    return Py_BuildValue("(Od)", ret, c_api_time);
}

static PyObject *countpairs_countpairs_wp(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//need not be used -> setting to NULL so any attempts to dereference will result in a crash. 
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif    
    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    double boxsize,pimax;
    int nthreads=1;
    char *binfile;
    size_t element_size;

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.instruction_set = -1;
    options.need_avg_sep = 0;
    options.periodic=1;
    options.c_api_timer = 0;
    static char *kwlist[] = {
        "boxsize",
        "pimax",
        "nthreads",
        "binfile",
        "X",
        "Y",
        "Z",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "output_rpavg",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX, SSE, FALLBACK */
        NULL
    };
    
    if( ! PyArg_ParseTupleAndKeywords(args, kwargs, "ddisO!O!O!|bbbi", kwlist,
                                      &boxsize,&pimax,&nthreads,&binfile,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &(options.verbose),
                                      &(options.need_avg_sep),
                                      &(options.c_api_timer),
                                      &(options.instruction_set))
        
        ){
        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");

        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In wp> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }
        
        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    options.boxsize=boxsize;

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }

    /* How many data points are there? And are they all of floating point type */
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return 
        Py_RETURN_NONE;
    }
    
    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = NULL, *y1_array = NULL, *z1_array = NULL;
    x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);
    
    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert input array to allowed floating point types (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        perror(NULL);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }


    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    
    results_countpairs_wp results;
    options.float_type = element_size;
    double c_api_time = 0.0;
    int status = countpairs_wp(ND1,X1,Y1,Z1,
                               boxsize,
                               nthreads,
                               binfile,
                               pimax,
                               &results,
                               &options, NULL);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);

    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    
#if 0
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        fprintf(stderr,"%lf %lf %lf %lf %"PRIu64"\n",results.rupp[i-1],results.rupp[i],rpavg,results.wp[i],results.npairs[i]);
    }
#endif

    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        PyObject *item = NULL;
        const double rpavg = results.rpavg[i];
        item = Py_BuildValue("(ddddk)", rlow,results.rupp[i],rpavg,results.wp[i],results.npairs[i]);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }
    free_results_wp(&results);
    return Py_BuildValue("(Od)", ret, c_api_time);
}


static PyObject *countpairs_countpairs_xi(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash. 
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif    

    PyArrayObject *x1_obj, *y1_obj, *z1_obj;
    double boxsize;
    int nthreads=4;
    char *binfile;

    static char *kwlist[] = {
        "boxsize",
        "nthreads",
        "binfile",
        "X",
        "Y",
        "Z",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "output_ravg",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX, SSE, FALLBACK */
        NULL
    };

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.periodic=1;
    options.instruction_set = -1; //from enum
    options.c_api_timer = 0;
    if( ! PyArg_ParseTupleAndKeywords(args, kwargs, "disO!O!O!|bbbi", kwlist,
                                      &boxsize,&nthreads,&binfile,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &(options.verbose),
                                      &(options.need_avg_sep),
                                      &(options.c_api_timer),
                                      &(options.instruction_set))
        ) {

        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");
        
        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In xi> Could not parse the arguments. Input parameters are: \n");

        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }
        
        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }

    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }
        


    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj, &element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return 
        Py_RETURN_NONE;
    }

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = NULL, *y1_array = NULL, *z1_array = NULL;
    x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert to array of allowed floating point type (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    results_countpairs_xi results;
    options.periodic = 1;
    options.float_type = element_size;
    double c_api_time=0.0;
    int status = countpairs_xi(ND1,X1,Y1,Z1,
                               boxsize,
                               nthreads,
                               binfile,
                               &results,
                               &options, NULL);
    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }
    NPY_END_THREADS;

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }


#if 0
    for(int i=1;i<results.nbin;i++) {
        const double rpavg = results.rpavg[i];
        fprintf(stderr,"%lf %lf %lf %lf %"PRIu64"\n",results.rupp[i-1],results.rupp[i],rpavg,results.xi[i],results.npairs[i]);
    }
#endif

    /* Build the output list */
    PyObject *ret = PyList_New(0);
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;i++) {
        PyObject *item = NULL;
        const double ravg = results.ravg[i];
        item = Py_BuildValue("(ddddk)", rlow,results.rupp[i],ravg,results.xi[i],results.npairs[i]);
        PyList_Append(ret, item);
        Py_XDECREF(item);
        rlow=results.rupp[i];
    }
    free_results_xi(&results);

    return Py_BuildValue("(Od)", ret, c_api_time);
}

static PyObject *countpairs_countspheres_vpf(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if PY_MAJOR_VERSION < 3
    (void) self;//to suppress the unused variable warning. Terrible hack
    PyObject *module = NULL;//should not be used -> setting to NULL so any attempts to dereference will result in a crash. 
#else
    //In python3, self is simply the module object that was returned earlier by init
    PyObject *module = self;
#endif    

    PyArrayObject *x1_obj=NULL, *y1_obj=NULL, *z1_obj=NULL;
    double rmax;
    int nbin,nc,num_pN;
    unsigned long seed=-1;
    static char *kwlist[] = {
        "rmax",
        "nbins",
        "nspheres",
        "num_pN",
        "seed",
        "X",
        "Y",
        "Z",
        "periodic",
        "verbose", /* keyword verbose -> print extra info at runtime + progressbar */
        "boxsize",
        "c_api_timer",
        "isa",/* instruction set to use of type enum isa; valid values are AVX, SSE, FALLBACK */
        NULL
    };

    struct config_options options = get_config_options();
    options.verbose = 0;
    options.periodic = 1;
    options.instruction_set = -1;
    options.c_api_timer = 0;
    if( ! PyArg_ParseTupleAndKeywords(args, kwargs,
                                      "diiikO!O!O!|bbdbi", kwlist,
                                      &rmax,&nbin,&nc,&num_pN,&seed,
                                      &PyArray_Type,&x1_obj,
                                      &PyArray_Type,&y1_obj,
                                      &PyArray_Type,&z1_obj,
                                      &(options.periodic),
                                      &(options.verbose),
                                      &(options.boxsize),
                                      &(options.c_api_timer),
                                      &(options.instruction_set))

        ) {

        PyObject_Print(kwargs, stdout, 0);
        fprintf(stdout, "\n");
        
        char msg[1024];
        int len=snprintf(msg, 1024,"ArgumentError: In vpf> Could not parse the arguments. Input parameters are: \n");
        
        /* How many keywords do we have? Subtract 1 because of the last NULL */
        const size_t nitems = sizeof(kwlist)/sizeof(*kwlist) - 1;
        int status = print_kwlist_into_msg(msg, 1024, len, kwlist, nitems);
        if(status != EXIT_SUCCESS) {
            fprintf(stderr,"Error message does not contain all of the keywords\n");
        }
        
        countpairs_error_out(module,msg);
        Py_RETURN_NONE;
    }
    /*This is for the fastest isa */
    if(options.instruction_set == -1) {
        options.instruction_set = highest_isa;
    }
    
    /* How many data points are there? And are they all of floating point type */
    size_t element_size;
    const int64_t ND1 = check_dims_and_datatype(module, x1_obj, y1_obj, z1_obj,&element_size);
    if(ND1 == -1) {
        //Error has already been set -> simply return 
        Py_RETURN_NONE;
    }

    /* Interpret the input objects as numpy arrays. */
    const int requirements = NPY_ARRAY_IN_ARRAY;
    PyObject *x1_array = NULL, *y1_array = NULL, *z1_array = NULL;
    x1_array = PyArray_FromArray(x1_obj, NOTYPE_DESCR, requirements);
    y1_array = PyArray_FromArray(y1_obj, NOTYPE_DESCR, requirements);
    z1_array = PyArray_FromArray(z1_obj, NOTYPE_DESCR, requirements);

    if (x1_array == NULL || y1_array == NULL || z1_array == NULL) {
        Py_XDECREF(x1_array);
        Py_XDECREF(y1_array);
        Py_XDECREF(z1_array);
        char msg[1024];
        snprintf(msg, 1024, "TypeError: In %s: Could not convert to array of allowed floating point type (doubles or floats). Are you passing numpy arrays?",
                 __FUNCTION__);
        countpairs_error_out(module, msg);
        Py_RETURN_NONE;
    }

    /* Get pointers to the data as C-types. */
    void *X1 = PyArray_DATA((PyArrayObject *) x1_array);
    void *Y1 = PyArray_DATA((PyArrayObject *) y1_array);
    void *Z1 = PyArray_DATA((PyArrayObject *) z1_array);

    /* Do the VPF calculation */
    results_countspheres results;
    options.float_type = element_size;
    double c_api_time=0.0;
    int status = countspheres(ND1, X1, Y1, Z1,
                              rmax, nbin, nc,
                              num_pN,
                              seed,
                              &results,
                              &options, NULL);

    if(options.c_api_timer) {
        c_api_time = options.c_api_time;
    }

    /* Clean up. */
    Py_DECREF(x1_array);Py_DECREF(y1_array);Py_DECREF(z1_array);
    if(status != EXIT_SUCCESS) {
        Py_RETURN_NONE;
    }

    /* Build the output list (of lists, since num_pN is determined at runtime) */
    PyObject *ret = PyList_New(0);
    const double rstep = rmax/(double)nbin ;
    for(int ibin=0;ibin<results.nbin;ibin++) {
        const double r=(ibin+1)*rstep;
        PyObject *item = PyList_New(0);
        PyObject *this_val = Py_BuildValue("d",r);
        PyList_Append(item, this_val);
        Py_XDECREF(this_val);
        for(int i=0;i<num_pN;i++) {
            this_val = Py_BuildValue("d",(results.pN)[ibin][i]);
            PyList_Append(item, this_val);
            Py_XDECREF(this_val);
        }
        PyList_Append(ret, item);
        Py_XDECREF(item);
    }

    free_results_countspheres(&results);
    return Py_BuildValue("(Od)", ret, c_api_time);
}
