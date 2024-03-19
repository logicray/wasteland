/*
 * print Celius-Fahrenheit table
 * for cel = 0, 20, ... ,300; floating-point version
 *
 */

#include<stdio.h>

main()
{
	float fahr;
	float celsius;
	float lower, upper, step;

	lower = 0;
	upper = 300;
	step = 20;

	celsius = lower;
	printf("cel \t fahr\n");
	while (celsius <= upper) {
		fahr = celsius * 9.0/5.0 + 32.0;
		printf("%3.0f %6.1f\n", celsius, fahr);
		celsius = celsius + step;
	}
}
