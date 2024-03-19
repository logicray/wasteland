/*************************************************************************
    > File Name: ./draw_freq_char.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月17日 星期二 00时11分17秒
	> draw a histogram of the frequencies of different characters 
 ************************************************************************/

#include<stdio.h>
main() {
	int c;
	int i;
	int alphabeta[26];
	for(i = 0; i < 26; i++)
		alphabeta[i] = 0;
	while( (c = getchar()) != EOF) {
		if( c >= 'a' && c <= 'z')
			alphabeta[c - 'a'] += 1;
	}
	for(i = 0; i < 26; i++){
		printf("%c:",i + 'a');
		int j;
		for(j = 0; j < alphabeta[i]; j++)
			printf("#");
		printf("\n");
	}
}
