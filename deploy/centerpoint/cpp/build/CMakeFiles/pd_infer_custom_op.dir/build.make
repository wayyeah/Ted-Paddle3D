# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yw/Paddle3D/deploy/centerpoint/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yw/Paddle3D/deploy/centerpoint/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/pd_infer_custom_op.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pd_infer_custom_op.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pd_infer_custom_op.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pd_infer_custom_op.dir/flags.make

CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o.depend
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o.cmake
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o: ../custom_ops/voxelize_op.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o"
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -E make_directory /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/.
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_voxelize_op.cu.o -D generated_cubin_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_voxelize_op.cu.o.cubin.txt -P /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o.cmake

CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o.depend
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o.cmake
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o: ../custom_ops/iou3d_nms_kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o"
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -E make_directory /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/.
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o -D generated_cubin_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o.cubin.txt -P /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o.cmake

CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o.depend
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o.cmake
CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o: ../custom_ops/postprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building NVCC (Device) object CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o"
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -E make_directory /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/.
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_postprocess.cu.o -D generated_cubin_file:STRING=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/./pd_infer_custom_op_generated_postprocess.cu.o.cubin.txt -P /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o.cmake

CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o: CMakeFiles/pd_infer_custom_op.dir/flags.make
CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o: ../custom_ops/voxelize_op.cc
CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o: CMakeFiles/pd_infer_custom_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o -MF CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o.d -o CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o -c /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/voxelize_op.cc

CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/voxelize_op.cc > CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.i

CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/voxelize_op.cc -o CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.s

CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o: CMakeFiles/pd_infer_custom_op.dir/flags.make
CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o: ../custom_ops/postprocess.cc
CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o: CMakeFiles/pd_infer_custom_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o -MF CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o.d -o CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o -c /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/postprocess.cc

CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/postprocess.cc > CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.i

CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yw/Paddle3D/deploy/centerpoint/cpp/custom_ops/postprocess.cc -o CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.s

# Object files for target pd_infer_custom_op
pd_infer_custom_op_OBJECTS = \
"CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o" \
"CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o"

# External object files for target pd_infer_custom_op
pd_infer_custom_op_EXTERNAL_OBJECTS = \
"/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o" \
"/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o" \
"/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o"

libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/custom_ops/voxelize_op.cc.o
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/custom_ops/postprocess.cc.o
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/build.make
libpd_infer_custom_op.so: /home/yw/anaconda3/envs/paddle/cuda-11.2/lib64/libcudart_static.a
libpd_infer_custom_op.so: /usr/lib/x86_64-linux-gnu/librt.so
libpd_infer_custom_op.so: CMakeFiles/pd_infer_custom_op.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library libpd_infer_custom_op.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pd_infer_custom_op.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pd_infer_custom_op.dir/build: libpd_infer_custom_op.so
.PHONY : CMakeFiles/pd_infer_custom_op.dir/build

CMakeFiles/pd_infer_custom_op.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pd_infer_custom_op.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pd_infer_custom_op.dir/clean

CMakeFiles/pd_infer_custom_op.dir/depend: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_iou3d_nms_kernel.cu.o
CMakeFiles/pd_infer_custom_op.dir/depend: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_postprocess.cu.o
CMakeFiles/pd_infer_custom_op.dir/depend: CMakeFiles/pd_infer_custom_op.dir/custom_ops/pd_infer_custom_op_generated_voxelize_op.cu.o
	cd /home/yw/Paddle3D/deploy/centerpoint/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yw/Paddle3D/deploy/centerpoint/cpp /home/yw/Paddle3D/deploy/centerpoint/cpp /home/yw/Paddle3D/deploy/centerpoint/cpp/build /home/yw/Paddle3D/deploy/centerpoint/cpp/build /home/yw/Paddle3D/deploy/centerpoint/cpp/build/CMakeFiles/pd_infer_custom_op.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pd_infer_custom_op.dir/depend

