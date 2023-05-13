/*************************************************************************
    > File Name: ./nonvisiabel_char_count.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月09日 星期一 23时56分21秒
 ************************************************************************/

#include<stdio.h>
main() {
	int nvc = 0;
	int c;
	while((c = getchar()) != EOF )
		if (c == '\n' || c == '\t' || c==' ')
			nvc++;
	printf("%d", nvc);
}
