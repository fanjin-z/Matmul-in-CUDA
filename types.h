#ifndef types_h
/* Do not change the code in this file, as doing so
 * could cause your submission to be graded incorrectly
 */

/*
 * Include this in every module that uses floating point
 * arithmetic, and declare all floating point values as "_DOUBLE_"
 * With a switch of a command line macro set up in the Makefile
 * we can then change the arithmetic
*/

#define types_h
#ifndef _DOUBLE
#define _DOUBLE_ float
#else
#define _DOUBLE_ double
#endif
#else
#endif
#include <stdint.h>
