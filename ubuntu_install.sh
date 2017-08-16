# System: Ubuntu LTS 16.04.3, Fresh install
# Options / Variables
#TODO: Makes tag for: Git code (spatialIF / 

CORES=6
MITSUBA=./build/release/binaries/mitsuba

# install all dependencies
sudo apt-get install cmake libboost-dev libboost-filesystem-dev libboost-thread-dev libqt4-dev libeigen3-dev libjpeg-dev libpng-dev libopenexr-dev libxerces-c-dev libglewmx-dev python-pip
pip install --upgrade pip

# Build the code
cd build
mkdir release
cd release
cmake ../..
make -j $CORES

# Go back to the project root
cd ../.. 

# Download scenes
wget https://dl.dropboxusercontent.com/u/37606091/research/2016_SpatialTF/scenes_spatialTF.tgz
tar -xvf scenes_spatialTF.tgz

# Test run Mitsuba
$MITSUBA -v 

# Install rgbe library
# This python library will be usefull for computing the metric, etc. 
# in order to reproduces the figures
git clone --recursive https://github.com/beltegeuse/rgbe.git
sudo pip install rgbe/

# Install production tools for Mitsuba
# This is to regenerate the results for the paper
git clone https://github.com/beltegeuse/mitsuba-prodTools.git





