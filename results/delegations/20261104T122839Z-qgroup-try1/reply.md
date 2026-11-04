```c
#include "qgroup.h"
#include <stdint.h>
#include <arm_neon.h>

/* Scalar fallback */
static float qgroup_dot_scalar(const uint32_t *w, const float *x, int n, int bits) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        uint32_t word_idx = i / (32 / bits);
        int bit_off = (i % (32 / bits)) * bits;
        uint32_t val;
        if (bits == 4) {
            uint8_t byte = ((const uint8_t *)w)[word_idx];
            if (i % 2 == 0) val = byte & 0x0F;
            else val = (byte >> 4) & 0x0F;
        } else {
            val = ((const uint8_t *)w)[word_idx];
        }
        sum += (float)val * x[i];
    }
    return sum;
}

/* NEON 4-bit implementation */
static float qgroup_dot_neon_4bit(const uint32_t *w, const float *x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i < n; i += 32) {
        uint8x16x2_t x_deinterleaved = vld2q_f32(x + i);
        uint8x16_t b = vld1q_u8((const uint8_t *)(w + i / 8));
        uint8x16_t lo = vandq_u8(b, vdupq_n_u8(0x0F));
        uint8x16_t hi = vshrq_n_u8(b, 4);
        uint16x8_t lo16 = vmovl_u8(vget_low_u8(lo));
        uint16x8_t hi16 = vmovl_u8(vget_low_u8(hi));
        uint32x4_t lo32 = vmovl_u16(vget_low_u16(lo16));
        uint32x4_t hi32 = vmovl_u16(vget_low_u16(hi16));
        float32x4_t lo_f = vcvtq_f32_u32(lo32);
        float32x4_t hi_f = vcvtq_f32_u32(hi32);
        acc0 = vfmaq_f32(acc0, lo_f, vget_low_f32(x_deinterleaved.val[0]));
        acc0 = vfmaq_f32(acc0, hi_f, vget_high_f32(x_deinterleaved.val[0]));
        acc1 = vfmaq_f32(acc1, lo_f, vget_low_f32(x_deinterleaved.val[1]));
        acc1 = vfmaq_f32(acc1, hi_f, vget_high_f32(x_deinterleaved.val[1]));
        acc2 = vfmaq_f32(acc2, lo_f, vget_low_f32(x_deinterleaved.val[0]));
        acc2 = vfmaq_f32(acc2, hi_f, vget_high_f32(x_deinterleaved.val[0]));
        acc3 = vfmaq_f32(acc3, lo_f, vget_low_f32(x_deinterleaved.val[1]));
        acc3 = vfmaq_f32(acc3, hi_f, vget_high_f32(x_deinterleaved.val[1]));
    }
    return vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
}

/* NEON 8-bit implementation */
static float qgroup_dot_neon_8bit(const uint32_t *w, const float *x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i < n; i += 16) {
        uint8x16_t b = vld1q_u8((const uint8_t *)(w + i / 4));
        uint16x8_t b16 = vmovl_u8(vget_low_u8(b));
        uint32x4_t b32 = vmovl_u16(vget_low_u16(b16));
        float32x4_t b_f = vcvtq_f32_u32(b32);
        float32x4_t x_f = vld1q_f32(x + i);
        acc0 = vfmaq_f32(acc0, b_f, x_f);
        b16 = vmovl_u8(vget_high_u8(b));
        b32 = vmovl_u16(vget_low_u16(b16));
        b_f = vcvtq_f32_u32(b32);
        x_f = vld1q_f32(x + i + 4);
        acc1 = vfmaq_f32(acc1, b_f, x_f);
    }
    return vaddvq_f32(acc0) + vaddvq_f32(acc1);
}

float qgroup_dot(const uint32_t *w, const float *x, int n, int bits) {
#if defined(__ARM_NEON)
    if (bits == 4) return qgroup_dot_neon_4bit(w, x, n);
    else if (bits == 8) return qgroup_dot_neon_8bit(w, x, n);
#endif
    return qgroup_dot_scalar(w, x, n, bits);
}
```