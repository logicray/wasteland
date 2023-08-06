#include <stdio.h>
#include <stdlib.h>

struct point
{
	int x;
	int y;
};

struct point  addpoint(struct point, struct point);

int main(int argc, char const *argv[])
{
	struct point a={3,8};
	struct point b={4,6};
	printf("%d\n", addpoint(a,b).x);
	printf("%d\n", a.x);
	return 0;
}


struct  point addpoint(struct point p1, struct point p2) {
	p1.x += p2.x;
	p1.y += p2.y;
    return p1;
}
