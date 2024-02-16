#!/bin/bash

# Check if the user is root, if not, exit
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
# Function to bring the background process into the foreground
bring_to_foreground() {
    fg %1
}

# Function to run a test for a specified duration
run_test() {
    test_name=$1
    duration=$2
    filename=$3
    echo "Running $test_name for $duration minute(s)..."
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp  /home/paam/Research/reference-system-paam/build/autoware_reference_system/$test_name > /home/paam/Research/data/V-B/"$filename" &
    test_pid=$!
    sleep "$duration"m
    kill "$test_pid"
    echo "$test_name completed."
}

# Set the ROS_LOCALHOST environment variable to 1
export ROS_LOCALHOST=1
ifconfig lo multicast
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Set the current environment to ../install/setup.bash
source /home/paam/Research/reference-system-paam/install/setup.bash
# Source ~/Research/ros2_galactic/install/setup.bash for the current script environment
source /home/paam/Research/ros2_galactic/install/setup.bash
export CYCLONEDDS_URI=file:///home/paam/Research/PAAM-RTAS/cyclonedds.xml

# Check if the ros2_daemon is running, if not, start it
if ! pgrep -x "ros2_daemon" > /dev/null; then
    echo "Starting ros2_daemon..."
    ros2 daemon start
fi
cd /home/paam/Research/reference-system-paam/

colcon build --symlink-install --packages-select rclcpp autoware_reference_system reference_system --cmake-args -DPICAS=TRUE -DPAAM=TRUE -DPAAM_PICAS=TRUE -DDIRECT_INVOCATION=FALSE
cd ./scripts/ 

# # Launch iox-roudi in the background
iox-roudi -c /home/paam/Research/reference-system-paam/roudi.toml &

# # Store the process ID of iox-roudi
iox_pid=$!



# # Launch PAAM server in the background
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp /home/paam/Research/reference-system-paam/build/paam_server/paam_server &
paam_pid=$!

# Wait for 15 seconds
sleep 15
# Run the tests that use picas and PAAM
run_test "autoware_default_singlethreaded_picas_multi_executors" 1 "singlethreaded_me_paam_picas.log"
run_test "autoware_default_multithreaded_picas" 1 "multithreaded_paam_picas.log"

#trap 'bring_to_foreground; kill $paam_pid' SIGINT
#trap 'bring_to_foreground; kill $iox_pid' SIGINT
pkill paam_server

cd /home/paam/Research/reference-system-paam/
# build the reference system with picas and direct invocation
colcon build --symlink-install --packages-select rclcpp autoware_reference_system reference_system --cmake-args -DPICAS=TRUE -DPAAM=FALSE -DPAAM_PICAS=TRUE -DDIRECT_INVOCATION=TRUE
cd ./scripts/ 

# Run the test that use picas and direct invocation
run_test "autoware_default_singlethreaded_picas_multi_executors" 1 "singlethreaded_me_di_picas.log"
run_test "autoware_default_multithreaded_picas" 1 "multithreaded_di_picas.log"

cd /home/paam/Research/reference-system-paam/
# Build the reference system without picas and paam 
colcon build --symlink-install --packages-select rclcpp autoware_reference_system reference_system --cmake-args -DPICAS=FALSE -DPAAM=FALSE -DPAAM_PICAS=FALSE -DDIRECT_INVOCATION=TRUE

cd ./scripts/ 
# Run the tests that use direct invocation and no picas
run_test "autoware_default_singlethreaded_multi_executors" 1 "singlethreaded_me_di_no_picas.log"
run_test "autoware_default_multithreaded" 1 "multithreaded_di_no_picas.log"

pkill iox-roudi

