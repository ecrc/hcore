# Installation

Installation requires `CMake` of version 3.2.3 at least. To build HCORE,
follow these instructions:

1.  Get HCORE from git repository

        git clone git@github.com:ecrc/hcore


2.  Go into hcore folder

        cd hcore

3.  Get ECRC's CMake Module as a submodule using git as follows.

        git submodule update --init --recursive

4.  Create build directory and go there

        mkdir build && cd build

5.  Use CMake to get all the dependencies

        cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/ 

6.  Build HCORE

        make -j

7.  Build local documentation (optional)

        make docs

8.  Install HCORE

        make install

9. Add line

        export PKG_CONFIG_PATH=/path/to/install:$PKG_CONFIG_PATH

    to your .bashrc file to use HCORE as a library.

Now you can use `pkg-config` executable to collect compiler and linker flags for HCORE.

# Library Usage

Before including `hcore.h` in the user code, 
`ARMPL` macro for ARM Performance libraries or `MKL` macro 
for Intel MKL should be defined for guiding `hcore.h` 
to include correct header file for the BLAS and LAPACK(E) libraries.

