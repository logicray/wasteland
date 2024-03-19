import cffi


ffi = cffi.FFI()

# 解析header文件并生成Python接口
ffi.cdef(
        "int sum(int x, int y);"
    )

# 包装动态链接库
lib = ffi.dlopen("./csrc/sum")

result = lib.sum(1, 2)
print(result)
