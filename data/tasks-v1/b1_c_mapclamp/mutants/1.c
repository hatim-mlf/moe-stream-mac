float map_clamped_f32(float x,float il,float ih,float ol,float oh) {
    return ol+(x-il)*(oh-ol)/(ih-il);
}
