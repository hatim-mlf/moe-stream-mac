int days_in_month(unsigned year, unsigned month) {
    static const int days[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (year < 1 || year > 9999 || month < 1 || month > 12) return 0;
    if (month == 2 && (year % 400u == 0 || (year % 4u == 0 && year % 100u != 0))) return 29;
    return days[month-1];
}
