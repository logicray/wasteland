/*************************************************************************
    > File Name: ./cel_fahr_function.c
    > Author: P.W
    > Created Time: 2017年01月17日 星期二 22时27分13秒
	> temperature conversion by function
 ************************************************************************/

#include<stdio.h>

float conver(float cel);

main() {
	float fahr;
	float celsius;
	float lower, upper, step;
	lower = 0;
	upper = 300;
	step = 20;
	for(celsius = lower; celsius < upper; celsius += step) {
		printf("%3.0f\t%6.3f \n", celsius, conver(celsius));
	}
}

float conver(float cel) {
	return cel * 9.0/5.0 + 32.0;
}
