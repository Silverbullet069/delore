
#include <stdio.h>

// Ground Truth: Vulnerabilities Detected:
// 1. Unvalidated input: The user input is not being validated, which could lead to a buffer overflow attack.
// 2. Insecure use of sleep(): The sleep() function is being used with user-supplied input, which could lead to a denial of service attack.

int i;
char inLine[64];
cin >> inLine;
i = atoi (inLine);
sleep(i);