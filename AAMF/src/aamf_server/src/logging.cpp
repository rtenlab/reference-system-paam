#include "logging.h"

aamf_logger::aamf_logger(){
    log_file.open(filename, std::ios::out | std::ios::app);
}

aamf_logger::~aamf_logger(){
    log_file.close();
}
aamf_logger::log(std::string message){
    log_file << message;
}