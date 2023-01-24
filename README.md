YuNet.API

## Conda install packages
    $ conda create -n face -c conda-forge python=3.10.8 conda-pack fastapi seqlog uvicorn[standard] numpy opencv
`conda-pack` save packages

## Build OpenCV GPU

### proxy
`export http_proxy="http://192.168.1.1:1" && export https_proxy="http://192.168.1.1:1"`  
### check cuda
```bash
nvcc --version
nvidia-smi
```
### export, useless
```bash
export python_exec=`which python`
export include_dir=`python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
export library=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`
export default_exec=`which python`

# export include_dir='/app/face/include'
# export library='/app/face/lib'

echo $python_exec
echo $include_dir
echo $library
echo $default_exec
```

### Install opencv dependencies
```bash
apt-get install build-essential cmake pkg-config
apt-get install libjpeg8-dev libtiff5-dev # libjasper-dev libpng12-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
apt-get install libatlas-base-dev gfortran
# one line
apt-get update && apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran -y
```

### Build
Links  
-   [How to verify CuDNN installation?](https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation)
-   [Build reference](https://danielhavir.github.io/notes/install-opencv/)
-   [Check Python interpreter and Library](https://stackoverflow.com/questions/64486389/cmake-could-not-find-pythonlibs-missing-python-include-dirs)
-   [deps](https://github.com/opencv/opencv/issues/18909)
-   [deps](https://github.com/open-mmlab/denseflow/blob/master/INSTALL.md)
-   [Build script reference](https://github.com/innerlee/setup/blob/master/zzopencv.sh)
-   [FFmpeg packages](https://launchpad.net/ubuntu/+source/ffmpeg)
-   [cuda Dockerfile repo](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.8.0/ubuntu2204/devel/cudnn8/Dockerfile)

download [opecv](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags)
```bash
cd opencv-xx
mkdir build && cd build
cmake \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=ON \
  -D PYTHON_LIBRARY=$(python3-config --prefix)/lib/libpython3.10.so \
  -D PYTHON_INCLUDE_DIR=$(python3-config --prefix)/include/python3.10 \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.7.0/modules \
  -D BUILD_EXAMPLES=ON ..
make -j"$(nproc)"
make install
ldconfig
ln -s /usr/local/lib/python3.10/site-packages/cv2 /opt/conda/envs/face/lib/python3.10/site-packages/cv2
```

finally, check cv2 cuda
```python
import cv2
print(cv2.getBuildInformation())
```

Logs
```
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/__init__.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/load_config_py2.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/load_config_py3.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/config.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/misc/__init__.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/misc/version.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/mat_wrapper/__init__.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/utils/__init__.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/gapi/__init__.py
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so
-- Set runtime path of "/usr/local/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so" to "/usr/local/lib:/usr/local/cuda/lib64"
-- Installing: /usr/local/lib/python3.10/site-packages/cv2/config-3.10.py
```

Errors
```
Could NOT find CUDNN (missing: CUDNN_LIBRARY CUDNN_INCLUDE_DIR) (Required is at least version "7.5")

dpkg -L libcudnn8

/usr/lib/x86_64-linux-gnu
/usr/include/cudnn.h

CMake Warning at cmake/OpenCVDetectPython.cmake:81 (message):
  CMake's 'find_host_package(PythonInterp 2.7)' found wrong Python version:

  PYTHON_EXECUTABLE=/opt/conda/envs/face/bin/python

  PYTHON_VERSION_STRING=3.10.8

  Consider providing the 'PYTHON2_EXECUTABLE' variable via CMake command line
  or environment variables

Call Stack (most recent call first):
  cmake/OpenCVDetectPython.cmake:271 (find_python)
  CMakeLists.txt:643 (include)


-- Could NOT find Python2 (missing: Python2_EXECUTABLE Interpreter) 
    Reason given by package:
        Interpreter: Wrong major version for the interpreter "/opt/conda/envs/face/bin/python"

-- Found PythonInterp: /opt/conda/envs/face/bin/python3 (found suitable version "3.10.8", minimum required is "3.2") 
-- Could NOT find PythonLibs (missing: PYTHON_INCLUDE_DIRS) (Required is exact version "3.10.8")
<string>:1: DeprecationWarning: 
```