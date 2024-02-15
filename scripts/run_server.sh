#! /bin/bash
# Check if the user is root, if not, exit
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
ifconfig lo multicast
export ROS_LOCALHOST=1
# Set the current environment to ../install/setup.bash
source ../install/setup.bash

# Source ~/Research/ros2_galactic/install/setup.bash for the current script environment
source /home/paam/Research/ros2_galactic/install/setup.bash

export CYCLONEDDS_URI=file:///home/paam/Research/PAAM-RTAS/cyclonedds.xml

# Launch iox-roudi in the background
iox-roudi -c ../roudi.toml &

# Store the process ID of iox-roudi
iox_pid=$!

# Function to bring the background process into the foreground
bring_to_foreground() {
    fg %1
}

# Launch PAAM server in the background
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ../build/paam_server/paam_server &
paam_pid=$!

# Wait for 15 seconds
sleep 15