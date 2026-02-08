#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    long long n = (argc > 1) ? atoll(argv[1]) : 1000000LL;
    long long sum = 0;

    for (long long i = 0; i < n; i++) {
        sum += (i % 7);
    }

    printf("n=%lld sum=%lld\n", n, sum);
    return 0;
}