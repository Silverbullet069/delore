
// auth.cpp
// Ground Truth: Vulnerabilities Detected:
// 1. Unchecked user input: The gets() function is used to read user input without any bounds checking, which can lead to a buffer overflow vulnerability.
// 2. Unknown input validation: The grantAccess() function is not properly validating user input, which can lead to an authentication bypass.

#include <stdio.h>

int main () {
    char username[8];
    int allow = 0;
    printf external link("Enter your username, please: ");
    gets(username);
    if (grantAccess(username)) {
        allow = 1;
    }
    if (allow != 0) {
        privilegedAction();
    }
    return 0;
}

// check_string.cpp
// Ground Truth: Vulnerabilities Detected:
// 1. Unchecked return value of sprintf() - The return value of sprintf() is not checked, which could lead to a buffer overflow.

#include <stdio.h>
#include <stdlib.h>
 
enum { BUFFER_SIZE = 10 };
 
int main() {
    char buffer[BUFFER_SIZE];
    int check = 0;
 
    sprintf(buffer, "%s", "This string does not meant anything ...");
 
    printf external link("check: %d", check);
 
    return EXIT_SUCCESS;
}


// dos.cpp
// Ground Truth: Vulnerabilities Detected:
// 1. Unvalidated input: The user input is not being validated, which could lead to a buffer overflow attack.
// 2. Insecure use of sleep(): The sleep() function is being used with user-supplied input, which could lead to a denial of service attack.

#include <stdio.h>

void dos() {
  //
int i;
char inLine[64];
cin >> inLine;
i = atoi (inLine);
sleep(i);
}   

// format_string_attack.cpp
// Ground truth: 
// 1. Unvalidated user input: The program does not check the length of the user input, which could lead to a buffer overflow attack.
// 2. Format string vulnerability: The program does not check the format of the user input, which could lead to a format string attack (An attacker crafts a malicious format string to read sensitive data from the application's memory, such as private keys or user credentials).

#include <stdio.h>
 
int main(int argc, char **argv) {
    printf(argv[1]);
 
    return 0;
}


// Ground Truth: No vulnerabilites founded in all of these functions
#include <stdio.h>

int add (int a, int b) {
  return a + b;
}

void nothing() {
  return;
}

void print_hello_world() {
  printf("Hello, World!");
}

// print_name
// Vulnerabilities Detected:
// 1. Lack of input validation - The scanf() function does not validate user input, which could lead to a buffer overflow attack.
// 2. Unknown input validation: The Sanitize() function's implementation is unknown, which could lead to a SQL injection attack if poor implementation.

#include <stdio.h>

int _tmain(int argc, _TCHAR* argv[])
{
	char name[64];
	printf("Enter your name: ");
	scanf("%s", name);
	Sanitize(name);
	printf("Welcome, %s!", name);
	return 0;
}


// Ground truth:
// Vulnerabilities Detected:
// 1. Out-of-bounds read: The memcpy() function is used to copy data from img.data to buff1 and buff2 without checking the size of the destination buffer, which could lead to an out-of-bounds read.
// 2. Out-of-bounds write: The buff3 and buff4 arrays are written to without checking the size of the source buffer, which could lead to an out-of-bounds write.
// 3. Uninitialized memory access: there are two uninitialized memory access variables, which could lead to undefined behavior.
// 4. Memory leak: The buff4 array is not freed if the size3 variable is greater than 10, which could lead to a memory leak.

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct Image
{
    char header[4];
    int width;
    int height;
    char data[10];
};

int ProcessImage(char* filename){

    FILE *fp;
    char ch;
    struct Image img;

    fp = fopen(filename,"r"); 

    if(fp == NULL)
    {
        printf("\nCan't open file or file doesn't exist.");
        exit(0);
    }

    printf("\n\tHeader\twidth\theight\tdata\t\r\n");

    while(fread(&img,sizeof(img),1,fp)>0){
        printf("\n\t%s\t%d\t%d\t%s\r\n",img.header,img.width,img.height,img.data);
    
        int size1 = img.width + img.height;
        char* buff1=(char*)malloc(size1);

        memcpy(buff1,img.data,sizeof(img.data));
        free(buff1);
    
        if (size1/2==0){
            free(buff1);
        }
        else{
            if(size1 == 123456){
                buff1[0]='a';
            }
        }

        int size2 = img.width - img.height+100;
        //printf("Size1:%d",size1);
        char* buff2=(char*)malloc(size2);

        memcpy(buff2,img.data,sizeof(img.data));

        int size3= img.width/img.height;
        //printf("Size2:%d",size3);

        char buff3[10];
        char* buff4 =(char*)malloc(size3);
        memcpy(buff4,img.data,sizeof(img.data));

        char OOBR_stack = buff3[size3+100];
        char OOBR_heap = buff4[100];

        buff3[size3+100]='c';
        buff4[100]='c';

        if(size3>10){
                buff4=0;
        }
        else{
            free(buff4);
        }

        free(buff2);
    }
    fclose(fp);
}

// sqlconnect.cpp
// Vulnerabilities Detected:
// 1. Unvalidated user input: argv[0] is not being validated and could be used to inject malicious code.
// 2. SQL injection: The SQLConnect function is vulnerable to SQL injection attacks.

#include <stdio.h>

int main(int argc, char *argv[])
{
	rc = SQLConnect(Example.ConHandle, argv[0], SQL_NTS,
	(SQLCHAR *) "", SQL_NTS, (SQLCHAR *) "", SQL_NTS);
}