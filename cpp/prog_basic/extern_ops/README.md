c and cpp cooperation


## postfix with `_a` is cpp code call c
build step 1
```
gcc -c calle_a.c -o calle_a.o
```

build step 2
```
g++ -o a.out caller_a.cpp calle_a.o
```

postfix with `_b` is c code call cpp

build step 1
```
g++ b_callee.cpp -fPIC -shared -o b_callee.so
```

build step 2
```
gcc b_caller.c b_callee.so -o b.out
```

then run `b.out`
