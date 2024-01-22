#include <iostream>
#include <fstream>
#include <string>
class paam_logger{
    public:
    paam_logger();
    ~paam_logger();
    void log(std::string message);
    std::ofstream log_file;
    char filename[255] = "/home/paam/Research/overhead/overhead.csv";
};
