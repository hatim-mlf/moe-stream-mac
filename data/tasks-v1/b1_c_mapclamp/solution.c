float map_clamped_f32(float x,float in_lo,float in_hi,float out_lo,float out_hi) {
    if (x<in_lo) x=in_lo;
    if (x>in_hi) x=in_hi;
    return out_lo+(x-in_lo)*(out_hi-out_lo)/(in_hi-in_lo);
}
