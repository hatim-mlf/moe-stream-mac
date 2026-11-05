#include <assert.h>
#include "../candidate.c"
int main(void) {
    assert(days_in_month(2024,2)==29 && days_in_month(2023,2)==28);
    assert(days_in_month(1900,2)==28 && days_in_month(2000,2)==29);
    assert(days_in_month(1,1)==31 && days_in_month(2020,13)==0 && days_in_month(0,1)==0);
    return 0;
}
