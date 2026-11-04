/* qgroup.h: inner kernel of qmat_matvec. Dot product of one group of packed quantized values with x.
 *
 *   returns sum_{j < n} q_j * x[j]
 *   q_j is the j-th UNSIGNED value packed in w: bits = 4 or 8, value j of the group sits in
 *   w[j / (32/bits)] at bit offset (j % (32/bits)) * bits, least significant first.
 *   On a little-endian CPU this means: byte k of the group holds values 2k (low nibble) and
 *   2k+1 (high nibble) for 4-bit, and value k for 8-bit.
 *   n is 32, 64 or 128. No validation (the caller validates).
 */
#ifndef QGROUP_H
#define QGROUP_H
#include <stdint.h>

float qgroup_dot(const uint32_t *w, const float *x, int n, int bits);

#endif
