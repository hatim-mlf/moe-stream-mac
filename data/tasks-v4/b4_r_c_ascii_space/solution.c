int is_ascii_space_byte(unsigned char value) {
    return value == 9u || value == 10u || value == 11u || value == 12u || value == 13u || value == 32u;
}
