# Accelerator Access Management Framework

### Packages:

1. paam_server_interfaces
2. paam_server
3. rclcpp (PiCAS scheduling)
4. autoware_reference_system (and dependencies
)
#### Dependencies

1. Ubuntu 20.04 LTS (Jetpack 5.1.1 for Jetson AGX Xavier 35.3.1)
2. [ROS2 Galactic](https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html)
3. GCC v9.3+
4. [CUDA 11.4+](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04) (Tested with 11.2-11.8)
5. Nvidia Driver 510.XX+(others may work, only tested with 510-535)
6. OpenCV 4.5.4
7. CV_Bridge
8. [CuDNN 8.2.4+](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
9. Boost
10. Tensorflow Lite

## Install Dependencies in Order

1. Uninstall any ROS2 prebuilt binaries (e.g. sudo apt remove ros-galactic*)
2. Uninstall libopencv-dev (sudo apt remove libopencv*)
3. Download and install Nvidia Drivers 
4. Download and install cuda 11.4+ 
5. Download and install CuDNN 8.6.0.166+
6. Download and install OpenCV 4.5.4 (Replace CUDA_ARCH_BIN=7.2 with the compute capability of your nvidia card; 7.2 is for Jetpack 5.1.1 on Xavier AGX)

Nvidia drivers, CUDA 11.4+, CuDNN 8.6.0.166+ could have been already installed as part of JetPack 5.1.1

```bash
sudo apt install libcudnn8-dev
mkdir -p ~/opencv_build
cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv/
git checkout 4.5.4
git pull origin 4.5.4
mkdir build
cd ../opencv_contrib
git checkout 4.5.4
git pull origin 4.5.4
cd ../opencv/build
```
Below change CUDNN_LIBRARY, CUDNN_INCLUDE_DIR, and OPENCV_EXTRA_MODULES_PATH for your system. 
For example, JetPack 5.1.1:
- -D CUDNN_LIBRARY=/lib/aarch64-linux-gnu/libcudnn.so
- -D CUDNN_INCLUDE_DIR=/usr/include
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D WITH_FFMPEG=ON \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D WITH_CUDNN=ON \
    -D CUDNN_LIBRARY=/usr/local/cuda/targets/aarch64-linux/lib/libcudnn.so \
    -D CUDNN_INCLUDE_DIR=/usr/local/cuda/targets/aarch64-linux/include \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_world=ON \
    -D CUDA_ARCH_BIN=7.2 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/home/<user_name>/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..
make -j$(nproc)
sudo make install
```
If cmake fails for V4L, install it: sudo apt-get install libv4l-dev

7. Download Ros2 Galactic and compile from source

Add the ROS 2 apt repository to your system
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-pip \
  python3-pytest-cov \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  wget

python3 -m pip install -U \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest \
  setuptools

mkdir -p ~/ros2_galactic/src
cd ~/ros2_galactic
vcs import --input https://raw.githubusercontent.com/ros2/ros2/galactic/ros2.repos src
sudo apt update
sudo apt upgrade
sudo rosdep init
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src --rosdistro galactic -y --skip-keys "fastcdr rti-connext-dds-5.3.1 urdfdom_headers"
sudo apt remove libopencv*
cd ~/ros2_galactic/
touch  ~/ros2_galactic/src/ros-visualization/qt_gui_core/qt_gui_cpp/COLCON_IGNORE
touch  ~/ros2_galactic/src/ros-visualization/rqt/rqt_gui_cpp/COLCON_IGNORE
vim ~/ros2_galactic/src/ros-visualization/rqt/rqt/setup.py
# Add the following lines: after version='1.1.2'
py_modules=[],

cd ~/ros2_galactic/src/ros-perception
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv
git checkout galactic
git pull origin galactic
cd ~/ros2_galactic/
colcon build --symlink-install

```
sudo rosdep init may fail because the file /etc/ros/rosdep/sources.list.d/20-default.list already exists. Delete it and run the command again. 

If python3-catkin-pkg-modules is not found, install using pip: pip3 install catkin-pkg-modules

## Configure Environment
Substitute 'cuda' with your cuda version if necessary, or create a symbolic link.

```bash
echo "source ~/ros2_galactic/setup.bash" >> ~/.bashrc
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >>  ~/.bashrc
source ~/.bashrc
```


## Building the package

1. Install Ros Galactic
2. Clone this workspace
3. Build the workspace

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPICAS=TRUE -DPAAM=TRUE
```

4. To build a specific package run:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPICAS=TRUE -DPAAM=TRUE --packages-select PACKAGE_NAME
```

## Running the example
In three separate terminals, as the *root* user, run the following to test the server
```bash
 iox-roudi -c /full/path/to/reference_system_paam/roudi.toml
```
```bash
source /root_dir/src/install/setup.bash
export CYCLONEDDS_URI=file:///full/path/to/reference_system_paam/cyclonedds.xml
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ./build/paam_server/paam_server
```
```bash
source /root_dir/src/install/setup.bash
export CYCLONEDDS_URI=file:///full/path/to/reference_system_paam/cyclonedds.xml
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ./build/test_package_name/test_package args

```
## Compile options:
-DPAAM - enables paam clients

-DOVERHEAD_DEBUG - enables logging for overhead

-DDIRECT_INVOCATION - enables the direct invocation of kernels

-DPICAS - enables the PiCAS scheduling of callbacks (required for PAAM server and rclcpp)

-DPAAM_PICAS - enables PiCAS scheduling of callbacks from clients using PAAM (allows PAAM server and rclcpp to compile and run with -DPICAS, but disables picas on clients)
 
 Note: compile options -DPAAM and -DDIRECT_INVOCATION should never be enabled simultaneously. 
 Note 2: -DPICAS is required to be enabled in rclcpp for the paam_server package. -- to test clients without PiCAS, it is preferred to set '-DPAAM_PICAS=FALSE' and recompile with PiCAS enabled '-DPICAS=TRUE'

## Running CTest and Generating Figures
```bash
sudo su
cd /repo-root/
source install/setup.bash
colcon build --symlink-install --cmake-args -DPICAS=TRUE -DPAAM=FALSE -D PAAM_PICAS=TRUE -DDIRECT_INVOCATION=TRUE -DRUN_BENCHMARK=ON -DTEST_PLATFORM=TRUE --packages-select autoware_reference_system
colcon test --packages-select autoware_reference_system
```
