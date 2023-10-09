#include <iostream>
#include <fstream>
#include <string>
class aamf_logger{
    public:
    aamf_logger();
    ~aamf_logger();
    void log(std::string message);
    std::ofstream log_file;
    char filename[255] = "/home/aamf/Research/overhead/overhead.csv";
};
