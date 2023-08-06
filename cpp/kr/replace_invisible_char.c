/*************************************************************************
    > File Name: replace_invisible_char.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月12日 星期四 21时49分39秒
	> copy input to output and replace tab with \t, backspace with \b, 
	> backslash with \\
 ************************************************************************/

#include<stdio.h>
main() {
	int c;
	while((c=getchar()) != EOF){
		if(c == '	')
			printf("%s","\\t");
		else if(c == '\n')
			printf("%s","\\b");
		else if(c == '\\')
			printf("%s","\\\\");
		else
			printf("%c",c);
	}
}
