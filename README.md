
# smart-mirror-facial-recognition

## Getting Started

### Setting up the Raspberry Pi for OpenCV
Using a Raspberry Pi 3 running Raspbian Stretch to install OpenCV Python release 3.4.1 with Python3.

#### Expand Filesystem (Optional)
If using brand new/default Raspbian Stretch, make space on microSD card using following instructions. 
```
sudo apt-get purge wolfram-engine
sudo apt-get purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove
```

#### Update Pi 
```
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
```
#### Install Dependencies 
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
```

#### Setup Python3 Development and pip Tools 
```
sudo apt-get install python3 python3-setuptools python3-dev
```
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```
#### Download OpenCV 3.4.1 and OpenCV-contrib 
```
cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.1.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip
unzip opencv.zip
unzip opencv_contrib.zip
```
#### Install Python Packages
```
sudo pip3 install numpy
sudo pip3 install imutils
sudo pip3 install sklearn
```
#### Build OpenCV 
```
cd ~/opencv-3.4.1/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.1/modules \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D BUILD_EXAMPLES=ON ..
```
#### Increase Swap Space 
Increase swap space from 100MB to 1024MB to facilitate compilation of OpenCV and prevent crashing on the Pi using nano editor. 
```
sudo nano -w /etc/dphys-swapfile 
```
Scroll down to "**CONF_SWAPSIZE**" and change value to 1024. Save file by using "**CTRL+O**" and then exit nano with "**CTRL+X**". 

Activate new swap space: 
```
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
```
#### Compile and Install OpenCV 
This process may take 2 to 3 hours. The Pi will get hot in the process, so be sure to provide cooling to the Pi. 
```
make -j4
```
Install OpenCV 
```
sudo make install
sudo ldconfig
```

#### Test Successful OpenCV Installation 
Open Python3 interpreter and enter the following to test successful installation. 
```
import cv2
cv2.__version__
```
The output should be '3.4.1'

#### Revert Swap Size and Free Up Space 
Similarly to increasing swap size, use the same commands to revert the swap size back to 100MB. Once reverted, restart the service. 
```
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start
```
Remove the downloaded zip files to free up space. 
```
cd ~
rm -rf opencv.zip opencv_contrib.zip
```

## Classes
There are several classes that can be ran independently from the app.
 - Camera
     - CV2 Camera Stream
     - Connects to device's camera and return frames
- Embedding
    - Processes dataset into embeddings file for model
- Enrollment
    - Captures faces and saves as dataset
- FacialDetection
    - Detects face given a frame and labels it
- Gesture
    - Detects a hand gesture and returns it's label
- TrainModel
    - Trains model with Embedding Data

### Testing the Classes
Each of the classes can be ran from the root folder through. The arguments for the class are defaulted and can be modified in the main function.
```
python scripts/class.py
```

## Application
The application is ran to handle the sign in, sign out and sign up of the user.
It is always running and is looking for an action gesture.

- If the user is not signed in and...
    - A five is detected the mirror will attempt to detect a face.
    - A two is detected the mirror will prompt the user to capture 5 pictures
    - The photo is captured by showing a five
- If the user is signed in...
    - The user is logged out by showing a zero
