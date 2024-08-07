## Makefile for C++ project using Boost
#
# @author Cedric "levif" Le Dillau
# @modified Aroli Marcellinus 
#
# Some notes:
# - Using ':=' instead of '=' assign the value at Makefile parsing time,
#   others are evaluated at usage time. This discards
# - Use ':set list' in Vi/Vim to show tabs (Ctrl-v-i force tab insertion)
#

# List to '.PHONY' all fake targets, those that are neither files nor folders.
# "all" and "clean" are good candidates.
.PHONY: all, clean

# Define the final program name
PROGNAME := drug_sim

# Pre-processor flags to be used for includes (-I) and defines (-D) 
CPPFLAGS := -I/usr/local/cuda/include
# CPPFLAGS :=
# CXX to set the compiler
# CXX := g++
CXX := nvcc
CXXLINK := nvlink

# CXXFLAGS is used for C++ compilation options.
#CXXFLAGS += -Wall -O0 -fpermissive -std=c++11
#CXXFLAGS += -Wall -O2 -fno-alias -fpermissive
# CXXFLAGS += -Wall
# Use this if you want to use Tomek 2019 cell model.
# Otherwise, comment it
#CXXFLAGS += -DTOMEK_2019

# LDFLAGS is used for linker (-g enables debug symbols)
# LDFLAGS  += -g -L/usr/local/cuda/lib64
LDFLAGS  += -g -L/usr/local/cuda/lib64 -arch=sm_70 -rdc=true

# List the project' sources to compile or let the Makefile recognize
# them for you using 'wildcard' function.
#
SOURCES	= $(wildcard *.cpp) $(wildcard **/*.cpp) $(wildcard *.c) $(wildcard **/*.c) $(wildcard **/*.cu) $(wildcard *.cu)

# List the project' headers or let the Makefile recognize
# them for you using 'wildcard' function.
#
HEADERS	= $(wildcard *.hpp) $(wildcard **/*.hpp) $(wildcard *.h) $(wildcard **/*.h) $(wildcard **/*.cuh) $(wildcard *.cuh)

# Construct the list of object files based on source files using
# simple extension substitution.
OBJECTS := $(SOURCES:%.cpp=%.o)
LIBS= -lopenblas -lpthread -lcudart -lcublas

#
# Now declare the dependencies rules and targets
#
# Starting with 'all' make it  becomes the default target when none 
# is specified on 'make' command line.
all : $(PROGNAME)

# Declare that the final program depends on all objects and the Makfile
$(PROGNAME) : $(OBJECTS) Makefile
	$(CXX) -o bin/$@ $(OBJECTS) $(LDFLAGS)

# Now the choice of using implicit rules or not (my choice)...
#
# Choice 1: use implicit rules and then we only need to add some dependencies
#           to each object.
#
## Tells make that each object file depends on all headers and this Makefile.
#$(OBJECTS) : $(HEADERS) Makefile
#
# Choice 2: don't use implicit rules and specify our will
%.o: %.cpp $(HEADERS) Makefile
	$(CXX) -x cu $(CXXFLAGS) $(CPPFLAGS) -dc -arch=sm_70 $(OUTPUT_OPTION) $<
# -dc -rdc=true 

# Simple clean-up target
# notes:
# - the '@' before 'echo' informs make to hide command invocation.
# - the '-' before 'rm' command to informs make to ignore errors.
clean :
	@echo "Clean."
	rm -rf *.o bin/$(PROGNAME)
	rm -rf **/*.o