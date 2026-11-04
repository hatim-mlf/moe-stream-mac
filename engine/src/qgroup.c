#include "qgroup.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>

static inline float32x4_t f4(uint32x4_t v) { return vcvtq_f32_u32(v); }

/* 4-bit: one block = 16 bytes = 32 values. lo[k] = value 2k, hi[k] = value 2k+1 (k = 0..15). */
static float dot4(const uint8_t *b8, const float *x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    for (int blk = 0; blk < n / 32; blk++) {
        const uint8x16_t b = vld1q_u8(b8 + 16 * blk);
        const uint8x16_t lo = vandq_u8(b, vdupq_n_u8(0x0F)), hi = vshrq_n_u8(b, 4);
        const uint16x8_t lo_a = vmovl_u8(vget_low_u8(lo)), lo_b = vmovl_high_u8(lo);   /* k 0-7, k 8-15 */
        const uint16x8_t hi_a = vmovl_u8(vget_low_u8(hi)), hi_b = vmovl_high_u8(hi);
        const float32x4_t ql[4] = { f4(vmovl_u16(vget_low_u16(lo_a))), f4(vmovl_high_u16(lo_a)),
                                    f4(vmovl_u16(vget_low_u16(lo_b))), f4(vmovl_high_u16(lo_b)) };  /* k 0-3,4-7,8-11,12-15 */
        const float32x4_t qh[4] = { f4(vmovl_u16(vget_low_u16(hi_a))), f4(vmovl_high_u16(hi_a)),
                                    f4(vmovl_u16(vget_low_u16(hi_b))), f4(vmovl_high_u16(hi_b)) };
        for (int h = 0; h < 4; h++) {
            const float32x4x2_t xv = vld2q_f32(x + 32 * blk + 8 * h);  /* val[0] = x[2k], val[1] = x[2k+1], k = 4h..4h+3 */
            acc0 = vfmaq_f32(acc0, ql[h], xv.val[0]);
            acc1 = vfmaq_f32(acc1, qh[h], xv.val[1]);
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

/* 8-bit: one block = 16 bytes = 16 values in order. */
static float dot8(const uint8_t *b8, const float *x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.f);
    for (int blk = 0; blk < n / 16; blk++) {
        const uint8x16_t b = vld1q_u8(b8 + 16 * blk);
        const uint16x8_t lo_a = vmovl_u8(vget_low_u8(b)), lo_b = vmovl_high_u8(b);
        const float32x4_t q[4] = { f4(vmovl_u16(vget_low_u16(lo_a))), f4(vmovl_high_u16(lo_a)),
                                   f4(vmovl_u16(vget_low_u16(lo_b))), f4(vmovl_high_u16(lo_b)) };
        for (int h = 0; h < 4; h++) {
            acc0 = vfmaq_f32(acc0, q[h], vld1q_f32(x + 16 * blk + 4 * h));
        }
    }
    return vaddvq_f32(acc0);
}

float qgroup_dot(const uint32_t *w, const float *x, int n, int bits) {
    const uint8_t *b8 = (const uint8_t *)w;   /* little-endian: byte order matches value order */
    return bits == 4 ? dot4(b8, x, n) : dot8(b8, x, n);
}

#else
float qgroup_dot(const uint32_t *w, const float *x, int n, int bits) {
    const int per = 32 / bits; const uint32_t mask = (1u << bits) - 1u; float s = 0.f;
    for (int j = 0; j < n; j++) s += (float)((w[j / per] >> ((j % per) * bits)) & mask) * x[j];
    return s;
}
#endif
