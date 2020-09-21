# Install

Openpose 설치

Ubuntu 16.04 LTS, anaconda3-4.2.0, python3.5, RTX2080, nvidia-430.64, CUDA 10.0
mujoco 150 pro, baseline, gym

** gcc-6버전 이상 권유 
anaconda 사용시:
$ conda install -c omgarcia gcc-6

** 그래픽 드라이버 관련
GTX 버전은 설치 해보지 않았지만 그대로 해도 무방할 것이라 생각
RTX 버전의 경우 computing architecture가 이전 버전이랑 다르기 때문에 CUDA 설정 시 다르게 변경해줘야 할 부분이 있음. 
기본 적으로 RTX2080은 7.5의 아키텍쳐를 사용함.
드라이버 설치는 아래 참조
https://nohboogy.tistory.com/2
https://nohboogy.tistory.com/3

CUDA 10.0 버전을 설치하기 위해선 NVIDIA 410 이상의 그래픽 드라이버가 설치되어야 함. 
430 버전의 경우 10.1과 호환되는 버전이긴 하지만 10.1을 설치했을 때 다른 오류가 많이 생겨 10.0으로 진행.

Ubuntu 설치 후 그래픽 드라이버가 이미 설치된 상황에서 sudo apt-get upgrade 명령어를 실행할 경우 커널이 업그레이드 되는 경우가 있음. 이 경우 재부팅 했을 때 해상도가 낮게 나오는데 다시 그래픽 드라이버를 설치하면 해결 됨.(CUDA를 다시 설치할 필요는 없음)

1. Opencv 4.2.0 설치
참조: https://qengineering.eu/install-caffe-on-ubuntu-18.04-with-opencv-4.2.html
caffe를 설치 하기 위해선 opencv source install이 필요함. 
하지만 anaconda 환경에서는 source build한 opencv python을 사용할 경우 library 링크 오류가 발생하기 때문에 이 opencv는 python에서 사용하는 용도가 아닌 caffe에 필요한 기본 프로그램으로서 설치 
opencv를 모두 설치한 후에
$ pip install opencv-python==4.2.0
$ pip install opencv-contrib-python
을 통해 python coding용 opencv를 설치해서 사용함.
(정석적인 방법은 아니고, 실행 시 warning이 뜨는 경우도 있지만 여지껏 유일하게 실행된 방법.)

1) dependencies 설치.

$ sudo apt-get update
$ sudo apt-get upgrade

$ sudo apt-get install build-essential cmake git unzip pkg-config
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
$ sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
$ sudo apt-get install libavresample-dev libvorbis-dev
$ sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
$ sudo apt-get install libgtk2.0-dev libcanberra-gtk*
$ sudo apt-get install x264 libxvidcore-dev libx264-dev libgtk-3-dev
$ sudo apt-get install python3-dev python3-numpy python3-pip
$ sudo apt-get install python3-testresources
$ sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
$ sudo apt-get install libv4l-dev v4l-utils
$ cd /usr/include/linux
$ sudo ln -s -f ../libv4l1-videodev.h videodev.h
$ cd ~
$ sudo apt-get install libxine2-dev
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
$ sudo apt-get update
$ sudo apt-get install libjasper-dev
$ sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
$ sudo apt-get install liblapack-dev gfortran
$ sudo apt-get install libhdf5-dev protobuf-compiler
$ sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev

2) opencv 다운로드

$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip

$ unzip opencv.zip
$ unzip opencv_contrib.zip

*선택사항
$ mv opencv-4.2.0 opencv
$ mv opencv_contrib-4.2.0 opencv_contrib

$ cd opencv
$ mkdir build
$ cd build

꼭 할 필요는 없지만 안하면 설치 시 디렉토리 지정을 해줘야함

3) opencv 설치

$ cd opencv
$ mkdir build
$ cd build

다음 명령어로 cmake
 
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_VTK=OFF \
        -D WITH_OPENGL=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=ON \
        -D WITH_NVCUVID=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_QT=ON \
        -D WITH_GTK=OFF \
        -D WITH_OPENGL=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_opencv_cudacodec=OFF \
        -D WITH_OPENMP=ON \
        -D CUDA_ARCH_BIN=5.3,6.0,6.1,7.0,7.5 \
        -D CUDA_ARCH_PTX=7.5\
        -D WITH_CUBLAS=ON ..

옵션 설명
옵션으로 바꾸지 않은 경우 디렉토리를 맞게 설정
Eigen이 설치되어 있지 않으면 libeigen3-dev_3.3.4-4_all.deb를 다운받아(폴더에 있음)
$ sudo dpkg -i libeigen3-dev_3.3.4-4_all.deb
후에 다시 cmake
QT 와 GTK가 같은 역할을 하므로 둘 중 하나만 ON
GTX2080을 사용할 경우 CUDA 아키텍쳐를 자동으로 설정 못하기 때문에 지정해둠. 아마 이전 버전의 그래픽 카드를 사용한다면 해당 옵션을 빼거나 그에 맞는 숫자를 써야 함. 

configuring이 잘 되었다면 

$ make -j[] ([]안에는 본인 컴퓨터 cpu core갯수를 입력)
$ sudo make install -j8
$ sudo gedit /etc/ld.so.conf.d/opencv.conf
다음을 추가
/usr/local/lib
$ sudo ldconfig
$ gedit ~/.bashrc

제일 아래쪽에 
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH

터미널에 
$ pkg-config --libs opencv4를 쳐 package library지정이 잘 되었는지 확인

$ pip install opencv-python==4.2.0.32
$ pip install opencv-contrib-python==4.2.0.32
$ python 
import cv2
cv2.__version__
import 및 버전 체크가 잘 되는 지 확인

2. Caffe 설치
1) openpose 3rdparty caffe 가져오기 
$ cd ~
$ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

설치된 openpose 디렉토리에서 
openpose/3rdparty/caffe 를 홈 디렉토리로 복사 (openpose build로 한번에 설치하려고 하면 오류 발생)
$ cp openpose/3rd/caffe /home/wooseok (wooseok은 본인 컴퓨터 이름으로)
기본 BVLC에서 caffe를 설치할 경우 clip layer 오류가 발생한다. openpose에서 이를 수정하여 올려 두었기 때문에 이 3rdparty caffe를 사용함.

2) caffe 설치
$ cd ~/caffe
$ cp Makefile.config.example Makefile.config
https://github.com/BVLC/caffe/compare/master...wooseokRo:master
위 링크를 보고 수정된 부분을 고친다. 
1. CMakeLists.txt에서 python 버전 변경 
2. Makefile에서 opencv4를 인식할 수 있도록 수정
3. Makefile.config에서 cuda 아키텍쳐 부분은 본인 그래픽카드에 맞는 설정으로 변경 & python 버전과 다른 라이브러리들 설정
4. cmake/Dependencies.cmake에서 HDF5 오류 제거를 위해 수정, boost 버전 설정(ubuntu 16.04에는 기본적으로 1.58버전이 설치되어 있을 것이다.)
5. opencv 버전 호환을 위해 
src/caffe/layers/window_data_layer.cpp
src/caffe/test/test_io.cpp
src/caffe/util/io.cpp
에서 CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_GRAYSCALE을 
cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE로 변경

(option) 6. GanHands API 사용을 위한 custom layer 추가 
CVPR2018_Model/proj_layer_caffe에 있는 코드를 복사 
1) heatmaps_from_vec_layer.hpp 을 ~caffe\include\caffe\layers로 복사
2) heatmaps_from_vec_layer.cpp 을 ~caffe\src\caffe\layers 로 복사 
3) ~caffe\src\caffe\proto\ caffe.proto 을 다음과 같이 수정 
 Inside the function message LayerParameter add this layer with correct next id:
 Example:
	// LayerParameter next available layer-specific ID: 148 (last added: recurrent_param)
	message LayerParameter {

	/* some standard layers */
	...
   optional HeatmapsFromVecParameter heatmaps_from_vec_layer = 147;
   
   }
    
4)	At the end of the file caffe.proto add:
	
	message HeatmapsFromVecParameter {
		optional uint32 heatmap_size = 1 [default = 32];
		}

$ mkdir build
$ cd build
$ cmake .. 
configuring이 잘 됐다면 pass
혹 ROS 3rdparty opencv나 이미 설치된 다른 opencv가 있다면 opencv가 다른 버전으로 잡혀 있을 수 있다. 
이때는 
cmake -DOpenCV_DIR=/home/wooseok/opencv/build .. (설치한 opencv 디렉토리의 build로 )
opencv 디렉토리 설정을 해주고 다시 cmake 한다. 

$ make -j8
$ make install -j8
$ make runtest -j8
$ make pycaffe -j8
$ make pytest -j8
여기까지 잘 진행됬다면 성공

$ gedit ~/.bashrc
맨 아래에 
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH}
export PYTHONPATH="${PYTHONPATH}:/home/wooseok/caffe/python"
$ source ~/.bashrc
$ python 
import caffe
caffe.__version__ 
import와 버전출력이 잘 되었다면 caffe설치 완료

3. Openpose 설치
1) openpose 설치
$ sudo apt-get update && sudo apt-get install build-essential freeglut3 freeglut3-dev libxmu-dev libxi-dev
$ cd ~/openpose
$ git checkout -b v1.5.1
$ mkdir build
$ cd build 
$ cmake ..
$ cmake-gui
(명령어로 바로 설정해도 되지만 gui로 하는 것이 더 편해서, 
설치 안되어 있다면 sudo apt-get install cmake-gui)

BUILD_PYTHON 체크
Caffe_INCLUDE_DIR   /caffe설치된 폴더/build/install/include 
Caffe_LIBS   /caffe 설치된 폴더/build/install/lib/libcaffe.so
OpenCV_DIR  /opencv 설치된 폴더/build

WITH_EIGEN 체크

configure & generate 후 나가기 
$ make -j8
$ make install -j8
$ sudo make install

설치가 완료 됐다면 
$ cd examples/tutorial_api_python  로 이동해서 
tutorial이 실행되는지 체크
$ gedit ~/.bashrc
맨 아래에 
export PYTHONPATH="${PYTHONPATH}:/home/wooseok/openpose/build/python"
$ source ~/.bashrc
$ python 
from openpose import pyopenpose as op
op.__file__
import와 file 위치 출력이 잘 됐다면 성공 

2) PyOpenpose 설치
$ cd ~
$ git clone https://github.com/FORTH-ModelBasedTracker/PyOpenPose
$ gedit ~/.bashrc
export OPENPOSE_ROOT=/home/wooseok/openpose (openpose 설치한 경로로 설정)
$ source ~/.bashrc
$ gedit CMakeLists.txt
50번째 줄 
- find_package(OpenCV 3 REQUIRED)
+ find_package(OpenCV 4 REQUIRED)

86, 87번째 줄 
-    set(CAFFE_INCLUDE_DIRS "$ENV{OPENPOSE_ROOT}/3rdparty/caffe/include")
-    set(CAFFE_LIBRARIES "$ENV{OPENPOSE_ROOT}/3rdparty/caffe/lib")
+    set(CAFFE_INCLUDE_DIRS "/home/wooseok/caffe/build/install/include")
+    set(CAFFE_LIBRARIES "/home/wooseok/caffe/build/install/lib")

$ mkdir build
$ cd build 
$ cmake -DWITH_PYTHON3=ON .. 
$ make -j8
$ sudo make install 

3) MonocularRGB_3D_Handpose_WACV18(3d position estimator) 설치
$ cd ~
$ git clone https://github.com/FORTH-ModelBasedTracker/MonocularRGB_3D_Handpose_WACV18
$ sudo apt install libgoogle-glog-dev libtbb-dev libcholmod3.0.6 libatlas-base-dev libopenni0 libbulletdynamics2.83.6

폴더에 있는 wacv18_libs_v1.0.tgz를 git clone한 폴더의 lib폴더에 압축해제 
ex) /home/wooseok/MonocularRGB_3D_Handpose_WACV18/lib
$ gedit ~/.bashrc
마지막에 다음을 추가 (본인 경로에 맞춰서)
export LD_LIBRARY_PATH=/home/wooseok/MonocularRGB_3D_Handpose_WACV18/lib/:$LD_LIBRARY_PATH

4. Pyrealsense 설치
$ pip install pyrealsense2


