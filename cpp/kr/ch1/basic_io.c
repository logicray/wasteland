/**
 * @file basic_io.c
 * @author logic 
 * @brief 
 * @version 0.1
 * @date 2021-12-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include<stdio.h>

main()
{
    int c;
    while ((c=getchar())!=EOF)
    {
        putchar(c);
    }
    

}