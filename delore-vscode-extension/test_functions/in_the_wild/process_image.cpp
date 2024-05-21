
// Ground truth:
// Vulnerabilities Detected:
// 1. Out-of-bounds read: The memcpy() function is used to copy data from img.data to buff1 and buff2 without checking the size of the destination buffer, which could lead to an out-of-bounds read.
// 2. Out-of-bounds write: The buff3 and buff4 arrays are written to without checking the size of the source buffer, which could lead to an out-of-bounds write.
// 3. Uninitialized memory access:  are accessed without being initialized, which could lead to undefined behavior.
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

int main(int argc,char **argv)
{
    ProcessImage(argv[1]);
}