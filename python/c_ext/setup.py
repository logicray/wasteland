from setuptools import setup, Extension

#setup(name="example_app", version="0.1", 
#      ext_modules=[Extension("example",["example.c"])])


extensions = []
c_ext = Extension("helloworld",
                  sources = ["./csrc/hello.c", "./csrc/example.c"])

extensions.append(c_ext)

setup(name="helloworld", version="0.1", 
      ext_modules=extensions,
      install_requires=[]
      )
