#include <stdio.h>

int main() {
    // print the first 10 Fibonacci numbers
    int a = 0, b = 1;

    for (int i = 0; i < 10; i++) {
        printf("%d\n", a);
        int c = a+b;
        a = b;
        b = c;
    }
}
