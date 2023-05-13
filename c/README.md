# cargo - pure c code practice
without any package build tool, just for practice
each package in src is independent.
you can compile each single package and run it.
read package details by their own readme doc.
if you want to run any single program

- kr package include some practise codes in reading  
'The C Programming Language' which written by Kernighan and Ritchie 
- algo package include some algorithm implements in c99
- step1. checkout code
```bash
git clone https://github.com/logic-pw/explore.git 
```
- step2. switch to corresponding dir
for example
```bash
cd ./c/kr/
```
- step3. compile some source code  with command,
```bash
gcc -o xxx.out xxx.c  //the xxx.c is source code name, and xxx.out is
executable file name.
```

- step4. run compiled program
```bash
./ myapp.out
```

compiled with clang12 or gcc9
