#include "logging.h"

paam_logger::paam_logger(){
    log_file.open(filename, std::ios::out | std::ios::app);
}

paam_logger::~paam_logger(){
    log_file.close();
}
paam_logger::log(std::string message){
    log_file << message;
}