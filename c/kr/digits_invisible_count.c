/*************************************************************************
    > File Name: ./digits_invisible_count.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月12日 星期四 23时40分16秒
	> count digits, white space and other characters
 ************************************************************************/

#include<stdio.h>
main(){
	int c, i, nwhite, nother;
	int ndigit[10];

	nwhite = nother = 0;
	/* initial array */
	for (i = 0; i < 10; ++i)
		ndigit[i] = 0;

	while( (c = getchar()) != EOF) {
		if (c >= '0' && c <= '9')
			++ndigit[c-'0'];
		else if (c == ' ' || c == '\n' || c == '\t')
			++nwhite;
		else
			++nother;
	}
		printf("digtis = ");
		for (i = 0; i < 10; ++i)
			printf(" %d", ndigit[i]);
		printf(", white space = %d, other = %d\n", nwhite, nother);
}
