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
include getData/CMakeFiles/collector.dir/depend.make

# Include the progress variables for this target.
include getData/CMakeFiles/collector.dir/progress.make

# Include the compile flags for this target's objects.
include getData/CMakeFiles/collector.dir/flags.make

getData/CMakeFiles/collector.dir/collectData.cpp.o: getData/CMakeFiles/collector.dir/flags.make
getData/CMakeFiles/collector.dir/collectData.cpp.o: ../getData/collectData.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sriram/projects/FinalIMD/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object getData/CMakeFiles/collector.dir/collectData.cpp.o"
	cd /home/sriram/projects/FinalIMD/build/getData && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/collector.dir/collectData.cpp.o -c /home/sriram/projects/FinalIMD/getData/collectData.cpp

getData/CMakeFiles/collector.dir/collectData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/collector.dir/collectData.cpp.i"
	cd /home/sriram/projects/FinalIMD/build/getData && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sriram/projects/FinalIMD/getData/collectData.cpp > CMakeFiles/collector.dir/collectData.cpp.i

getData/CMakeFiles/collector.dir/collectData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/collector.dir/collectData.cpp.s"
	cd /home/sriram/projects/FinalIMD/build/getData && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sriram/projects/FinalIMD/getData/collectData.cpp -o CMakeFiles/collector.dir/collectData.cpp.s

getData/CMakeFiles/collector.dir/collectData.cpp.o.requires:
.PHONY : getData/CMakeFiles/collector.dir/collectData.cpp.o.requires

getData/CMakeFiles/collector.dir/collectData.cpp.o.provides: getData/CMakeFiles/collector.dir/collectData.cpp.o.requires
	$(MAKE) -f getData/CMakeFiles/collector.dir/build.make getData/CMakeFiles/collector.dir/collectData.cpp.o.provides.build
.PHONY : getData/CMakeFiles/collector.dir/collectData.cpp.o.provides

getData/CMakeFiles/collector.dir/collectData.cpp.o.provides.build: getData/CMakeFiles/collector.dir/collectData.cpp.o

# Object files for target collector
collector_OBJECTS = \
"CMakeFiles/collector.dir/collectData.cpp.o"

# External object files for target collector
collector_EXTERNAL_OBJECTS =

bin/collector: getData/CMakeFiles/collector.dir/collectData.cpp.o
bin/collector: getData/CMakeFiles/collector.dir/build.make
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libicubmod.a
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_OS.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_sig.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_math.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_dev.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_name.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_init.so
bin/collector: /usr/local/lib/libopencv_videostab.so.2.4.9
bin/collector: /usr/local/lib/libopencv_video.so.2.4.9
bin/collector: /usr/local/lib/libopencv_ts.a
bin/collector: /usr/local/lib/libopencv_superres.so.2.4.9
bin/collector: /usr/local/lib/libopencv_stitching.so.2.4.9
bin/collector: /usr/local/lib/libopencv_photo.so.2.4.9
bin/collector: /usr/local/lib/libopencv_ocl.so.2.4.9
bin/collector: /usr/local/lib/libopencv_objdetect.so.2.4.9
bin/collector: /usr/local/lib/libopencv_nonfree.so.2.4.9
bin/collector: /usr/local/lib/libopencv_ml.so.2.4.9
bin/collector: /usr/local/lib/libopencv_legacy.so.2.4.9
bin/collector: /usr/local/lib/libopencv_imgproc.so.2.4.9
bin/collector: /usr/local/lib/libopencv_highgui.so.2.4.9
bin/collector: /usr/local/lib/libopencv_gpu.so.2.4.9
bin/collector: /usr/local/lib/libopencv_flann.so.2.4.9
bin/collector: /usr/local/lib/libopencv_features2d.so.2.4.9
bin/collector: /usr/local/lib/libopencv_core.so.2.4.9
bin/collector: /usr/local/lib/libopencv_contrib.so.2.4.9
bin/collector: /usr/local/lib/libopencv_calib3d.so.2.4.9
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libcartesiancontrollerserver.a
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libcartesiancontrollerclient.a
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libiKin.a
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libctrlLib.a
bin/collector: /usr/lib/libgsl.so
bin/collector: /usr/lib/libgslcblas.so
bin/collector: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libipopt.so
bin/collector: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinmumps.so
bin/collector: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinlapack.so
bin/collector: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinmetis.so
bin/collector: /home/sriram/Documents/Ipopt-3.11.9/build/lib/libcoinblas.so
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libgazecontrollerclient.a
bin/collector: /home/sriram/Documents/icub-main-1.1.15/build/lib/libdebugInterfaceClient.a
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_math.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_dev.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_name.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_init.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_sig.so
bin/collector: /home/sriram/Documents/yarp-2.3.63/build/lib/libYARP_OS.so
bin/collector: /usr/local/lib/libopencv_nonfree.so.2.4.9
bin/collector: /usr/local/lib/libopencv_ocl.so.2.4.9
bin/collector: /usr/local/lib/libopencv_gpu.so.2.4.9
bin/collector: /usr/local/lib/libopencv_photo.so.2.4.9
bin/collector: /usr/local/lib/libopencv_objdetect.so.2.4.9
bin/collector: /usr/local/lib/libopencv_legacy.so.2.4.9
bin/collector: /usr/local/lib/libopencv_video.so.2.4.9
bin/collector: /usr/local/lib/libopencv_ml.so.2.4.9
bin/collector: /usr/local/lib/libopencv_calib3d.so.2.4.9
bin/collector: /usr/local/lib/libopencv_features2d.so.2.4.9
bin/collector: /usr/local/lib/libopencv_highgui.so.2.4.9
bin/collector: /usr/local/lib/libopencv_imgproc.so.2.4.9
bin/collector: /usr/local/lib/libopencv_flann.so.2.4.9
bin/collector: /usr/local/lib/libopencv_core.so.2.4.9
bin/collector: getData/CMakeFiles/collector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/collector"
	cd /home/sriram/projects/FinalIMD/build/getData && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/collector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
getData/CMakeFiles/collector.dir/build: bin/collector
.PHONY : getData/CMakeFiles/collector.dir/build

getData/CMakeFiles/collector.dir/requires: getData/CMakeFiles/collector.dir/collectData.cpp.o.requires
.PHONY : getData/CMakeFiles/collector.dir/requires

getData/CMakeFiles/collector.dir/clean:
	cd /home/sriram/projects/FinalIMD/build/getData && $(CMAKE_COMMAND) -P CMakeFiles/collector.dir/cmake_clean.cmake
.PHONY : getData/CMakeFiles/collector.dir/clean

getData/CMakeFiles/collector.dir/depend:
	cd /home/sriram/projects/FinalIMD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sriram/projects/FinalIMD /home/sriram/projects/FinalIMD/getData /home/sriram/projects/FinalIMD/build /home/sriram/projects/FinalIMD/build/getData /home/sriram/projects/FinalIMD/build/getData/CMakeFiles/collector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : getData/CMakeFiles/collector.dir/depend

