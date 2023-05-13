/*************************************************************************
    > File Name: ./power.c
    > Author: P.W
    > Created Time: 2017年01月17日 星期二 21时40分35秒
 ************************************************************************/

#include<stdio.h>

int power(int m, int n);

main()
{
	int i;
	for(i = 0; i < 10; ++i)
		printf("%d %d %d\n", i, power(2, i), power(-3, i));
	return 0;
}

/* raise base to n-th power; n >= 0 */
int power(int base, int n)
{
	int i, p;

	p = 1;
	for( i = 1; i <= n; ++i)
		p = p * base;
	return p;
}
