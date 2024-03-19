/*
 *
 *
 */
#include<stdio.h>

#define IN 1
#define OUT 0

main() {
    int c;
    int length[5];
    int i;
    i = 0;
    int count;
    count = 0;
    while( (c = getchar()) != EOF){
        if(c != ' ' && c != '\n' && c != '\t') {
            count ++;
        }
        else  {
            length[i] = count;
            count = 0;
            i++;
        }
    }
    /*printf("%d \n",length[0]);*/
    int j;
    for(i = 0; i < 5; i++) {
        /* printf("%d \t",length[i]); */
        for(j = 0; j < length[i]; j++)
            printf("#");
        printf("\n");
    }
}
