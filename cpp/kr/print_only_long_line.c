/*************************************************************************
    > File Name: ./longest_line.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月19日 星期四 21时50分33秒
	> print the  line which longer than 80 characters
 ************************************************************************/

#include<stdio.h>
#define MAXLINE 1000  /*maxium input line length */
#define INTERCEPT 20  /* if line length > intercept ,print theline */

int getline(char line[], int maxline);
void copy(char to[], char from[]);

main()
{
	int len;
	int max;
	char line[MAXLINE];
	char longest[MAXLINE];
	
	max = 0;
	while ((len = getline(line, MAXLINE)) > 0)
		if (len > INTERCEPT) {
			printf("\n %s", line);
		}
	return 0;
}

int getline(char s[], int lim)
{
	int c, i;

	for (i = 0; i < lim-1 && (c=getchar()) != EOF && c != '\n'; i++)
		s[i] = c;
	if (c == '\n') {
		s[i] = c;
		++i;
	}
	s[i] = '\0';
	return i;
}

/* copy from to to;*/
void copy(char to[], char from[])
{
	int i;

	i = 0;
	while(( to[i] = from[i]) != '\0')
		++i;
}

