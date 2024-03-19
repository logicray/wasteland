/*************************************************************************
    > File Name: ./longest_line.c
    > Author: P.W
    > Mail: mangobada@163.com 
    > Created Time: 2017年01月19日 星期四 21时50分33秒
	> print the longest input line
 ************************************************************************/

#include<stdio.h>
#define MAXLINE 10  /*maxium input line length */

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
		if (len > max) {
			max = len;
			copy(longest, line);
		}
	if (max > 0)
		printf("\n %d %s", max, longest);
	return 0;
}

int getline(char s[], int lim)
{
	int c, i ,j;
	j = 0;

	for (i = 0; (c=getchar()) != EOF && c != '\n'; i++)
		if ( i < lim){
			s[j] = c;
			++j;
		}
	if (c == '\n') {
		s[j] = c;
		++j;
		++i;
	}
	s[j] = '\0';
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

