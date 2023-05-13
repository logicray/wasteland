/*************************************************************************
    > File Name: ./one_word_per_line.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月12日 星期四 23时05分20秒
	> print input one word per line to output
 ************************************************************************/

#include<stdio.h>
main() {
	int c;
	while((c = getchar()) != EOF){
		if(c == ' ' || c == '\t' || c == '\n')
			printf("\n");
		else
			printf("%c",c);
	}
}
