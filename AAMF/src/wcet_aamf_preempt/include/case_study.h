#include <functional>
#include <memory>
#include <thread>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>  //might need cmakelist update
#include <sys/time.h>
#include <unistd.h>  //might need cmakelist update
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.
#include <boost/lexical_cast.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_layout.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>
#include "rclcpp/rclcpp.hpp"
#include "aamf_server_interfaces/msg/gpu_request.hpp" 
#include "aamf_server_interfaces/msg/gpu_register.hpp"
#include "aamf_structs.h"
