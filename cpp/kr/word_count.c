/*************************************************************************
    > File Name: ./word_count.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月12日 星期四 22时39分09秒
	> count lines, words. and characters in input
 ************************************************************************/

#include<stdio.h>
 
#define IN 1 /*inside a word*/
#define OUT 0 /* outside a word */

main(){
	int c, nl, nw, nc, state;
	state = OUT;
	nl = nw = nc = 0;
	while ( (c = getchar()) != EOF) {
		++nc;
		if (c == '\n')
			++nl;
		if (c ==' ' || c == '\n' || c == '\t')
			state = OUT;
		else if( state == OUT) {
			state = IN;
			++nw;
		}
	}
	printf("%d %d %d\n", nl, nw, nc);
}
