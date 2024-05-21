
// Ground truth: 
// 1. Unvalidated user input: The program does not check the length of the user input, which could lead to a buffer overflow attack.
// 2. Format string vulnerability: The program does not check the format of the user input, which could lead to a format string attack (An attacker crafts a malicious format string to read sensitive data from the application's memory, such as private keys or user credentials).

#include <stdio.h>
 
int main(int argc, char **argv) {
    printf(argv[1]);
 
    return 0;
}