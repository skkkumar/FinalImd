# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sriram/projects/FinalIMD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sriram/projects/FinalIMD/build

# Include any dependencies generated for this target.
include TrainML/CMakeFiles/train.dir/depend.make

# Include the progress variables for this target.
include TrainML/CMakeFiles/train.dir/progress.make

# Include the compile flags for this target's objects.
include TrainML/CMakeFiles/train.dir/flags.make

TrainML/CMakeFiles/train.dir/train.cpp.o: TrainML/CMakeFiles/train.dir/flags.make
TrainML/CMakeFiles/train.dir/train.cpp.o: ../TrainML/train.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sriram/projects/FinalIMD/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object TrainML/CMakeFiles/train.dir/train.cpp.o"
	cd /home/sriram/projects/FinalIMD/build/TrainML && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/train.dir/train.cpp.o -c /home/sriram/projects/FinalIMD/TrainML/train.cpp

TrainML/CMakeFiles/train.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train.dir/train.cpp.i"
	cd /home/sriram/projects/FinalIMD/build/TrainML && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sriram/projects/FinalIMD/TrainML/train.cpp > CMakeFiles/train.dir/train.cpp.i

TrainML/CMakeFiles/train.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train.dir/train.cpp.s"
	cd /home/sriram/projects/FinalIMD/build/TrainML && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sriram/projects/FinalIMD/TrainML/train.cpp -o CMakeFiles/train.dir/train.cpp.s

TrainML/CMakeFiles/train.dir/train.cpp.o.requires:
.PHONY : TrainML/CMakeFiles/train.dir/train.cpp.o.requires

TrainML/CMakeFiles/train.dir/train.cpp.o.provides: TrainML/CMakeFiles/train.dir/train.cpp.o.requires
	$(MAKE) -f TrainML/CMakeFiles/train.dir/build.make TrainML/CMakeFiles/train.dir/train.cpp.o.provides.build
.PHONY : TrainML/CMakeFiles/train.dir/train.cpp.o.provides

TrainML/CMakeFiles/train.dir/train.cpp.o.provides.build: TrainML/CMakeFiles/train.dir/train.cpp.o

# Object files for target train
train_OBJECTS = \
"CMakeFiles/train.dir/train.cpp.o"

# External object files for target train
train_EXTERNAL_OBJECTS =

bin/train: TrainML/CMakeFiles/train.dir/train.cpp.o
bin/train: TrainML/CMakeFiles/train.dir/build.make
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libicubmod.a
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_OS.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_sig.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_math.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_dev.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_name.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_init.so
bin/train: /usr/local/lib/libopencv_videostab.so.2.4.9
bin/train: /usr/local/lib/libopencv_video.so.2.4.9
bin/train: /usr/local/lib/libopencv_ts.a
bin/train: /usr/local/lib/libopencv_superres.so.2.4.9
bin/train: /usr/local/lib/libopencv_stitching.so.2.4.9
bin/train: /usr/local/lib/libopencv_photo.so.2.4.9
bin/train: /usr/local/lib/libopencv_ocl.so.2.4.9
bin/train: /usr/local/lib/libopencv_objdetect.so.2.4.9
bin/train: /usr/local/lib/libopencv_nonfree.so.2.4.9
bin/train: /usr/local/lib/libopencv_ml.so.2.4.9
bin/train: /usr/local/lib/libopencv_legacy.so.2.4.9
bin/train: /usr/local/lib/libopencv_imgproc.so.2.4.9
bin/train: /usr/local/lib/libopencv_highgui.so.2.4.9
bin/train: /usr/local/lib/libopencv_gpu.so.2.4.9
bin/train: /usr/local/lib/libopencv_flann.so.2.4.9
bin/train: /usr/local/lib/libopencv_features2d.so.2.4.9
bin/train: /usr/local/lib/libopencv_core.so.2.4.9
bin/train: /usr/local/lib/libopencv_contrib.so.2.4.9
bin/train: /usr/local/lib/libopencv_calib3d.so.2.4.9
bin/train: /home/sriram/Documents/GURLS/build/lib/libgurls++.a
bin/train: /home/sriram/Documents/GURLS/build/external/lib/libopenblas.a
bin/train: /home/sriram/Documents/GURLS/build/external/lib/libboost_serialization-mt.a
bin/train: /home/sriram/Documents/GURLS/build/external/lib/libboost_date_time-mt.a
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libcartesiancontrollerserver.a
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libcartesiancontrollerclient.a
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libiKin.a
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libctrlLib.a
bin/train: /usr/lib/libgsl.so
bin/train: /usr/lib/libgslcblas.so
bin/train: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libipopt.so
bin/train: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinmumps.so
bin/train: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinlapack.so
bin/train: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinmetis.so
bin/train: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinblas.so
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libgazecontrollerclient.a
bin/train: /home/sriram/Documents/icub-main-1.1.15/build/lib/libdebugInterfaceClient.a
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_math.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_dev.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_name.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_init.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_sig.so
bin/train: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_OS.so
bin/train: /usr/local/lib/libopencv_nonfree.so.2.4.9
bin/train: /usr/local/lib/libopencv_ocl.so.2.4.9
bin/train: /usr/local/lib/libopencv_gpu.so.2.4.9
bin/train: /usr/local/lib/libopencv_photo.so.2.4.9
bin/train: /usr/local/lib/libopencv_objdetect.so.2.4.9
bin/train: /usr/local/lib/libopencv_legacy.so.2.4.9
bin/train: /usr/local/lib/libopencv_video.so.2.4.9
bin/train: /usr/local/lib/libopencv_ml.so.2.4.9
bin/train: /usr/local/lib/libopencv_calib3d.so.2.4.9
bin/train: /usr/local/lib/libopencv_features2d.so.2.4.9
bin/train: /usr/local/lib/libopencv_highgui.so.2.4.9
bin/train: /usr/local/lib/libopencv_imgproc.so.2.4.9
bin/train: /usr/local/lib/libopencv_flann.so.2.4.9
bin/train: /usr/local/lib/libopencv_core.so.2.4.9
bin/train: TrainML/CMakeFiles/train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/train"
	cd /home/sriram/projects/FinalIMD/build/TrainML && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
TrainML/CMakeFiles/train.dir/build: bin/train
.PHONY : TrainML/CMakeFiles/train.dir/build

TrainML/CMakeFiles/train.dir/requires: TrainML/CMakeFiles/train.dir/train.cpp.o.requires
.PHONY : TrainML/CMakeFiles/train.dir/requires

TrainML/CMakeFiles/train.dir/clean:
	cd /home/sriram/projects/FinalIMD/build/TrainML && $(CMAKE_COMMAND) -P CMakeFiles/train.dir/cmake_clean.cmake
.PHONY : TrainML/CMakeFiles/train.dir/clean

TrainML/CMakeFiles/train.dir/depend:
	cd /home/sriram/projects/FinalIMD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sriram/projects/FinalIMD /home/sriram/projects/FinalIMD/TrainML /home/sriram/projects/FinalIMD/build /home/sriram/projects/FinalIMD/build/TrainML /home/sriram/projects/FinalIMD/build/TrainML/CMakeFiles/train.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : TrainML/CMakeFiles/train.dir/depend
