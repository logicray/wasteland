/*
 * implement a stack in c
 */

#include<stdio.h>
#include<stdlib.h>

#define ElementType int
#define MAXSIZE 1024

typedef struct {
    ElementType data[MAXSIZE];
	int top;
}Stack;

Stack* initStack(){
	Stack* stack;
	stack = (Stack*)malloc(sizeof(Stack));
	if(!stack) {
		printf("no space");
		return NULL;
	}
	stack->top = -1;
	return stack;
}

int isFull(Stack* stack) {
	if (stack->top == MAXSIZE - 1) {
		printf("stack is full\n");
		return 1;
	}
    return 0;
}

int isEmpty(Stack *stack){
	if (stack->top == -1){
		printf("stack is empty\n");
		return 1;
	}
	return 0;
}

void push(Stack* stack, ElementType item) {
	if (isFull(stack)) {
		return;
	}
	stack->data[++stack->top] = item;
}

ElementType pop(Stack* stack) {
	if (isEmpty(stack)) {
		return -1;
	}
	return stack->data[stack->top--];
}

void printStack(Stack* stack) {
	printf("the elements of stack are: ");
	int i;
	for (i = stack->top; i >= 0; i--) {
		printf("%d ", stack->data[i]);
	}
	printf("\n");
}

int main(int arg, const char * argn[]) {
	Stack* stack;
	stack = initStack();
	push(stack,1);
	push(stack,2);
	push(stack,3);
	printStack(stack);
	pop(stack);
	pop(stack);
	printStack(stack);
	return 0;
}
