/*************************************************************************
    > File Name: ./strlen.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年04月16日 星期日 22时50分17秒
 ************************************************************************/

#include<stdio.h>

int mystrlen(char s[])
{
	int i;
	while(s[i] != '\0')
		++i;
	return i;
}

main(){
    printf("%d", mystrlen("frewgtw"));
}
