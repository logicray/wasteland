/*************************************************************************
    > File Name: ./remove_multi_space.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月10日 星期二 00时04分07秒
	> remove redundant spaces if there are more than one
 ************************************************************************/

#include<stdio.h>
main() {
	int last = 0;
	int c;
	while( (c = getchar()) != EOF) {
		if(last == c && c == ' ')
			;
		else
			printf("%c",c);
		last = c;
	}
}

