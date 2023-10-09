#ifndef ADMISSIONS_CONTROL
#define ADMISSIONS_CONTROL
#endif

#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>
#include <algorithm>
#include <memory>

class callback;
class cpu;
class executor;
class chain;
class chainset;
struct timer_callback;
class callback_row;
class accelerator;

class accelerator
{
public:
    int get_id(void);
    double get_util(void);
    accelerator(void);
    accelerator(int id, double util, std::string type);
    accelerator(const accelerator &second);
    accelerator operator=(const accelerator &other);
    void swap(accelerator &first, accelerator &second);
    void set_util(double util);
    void set_id(int id);
    std::string get_type(void);
    void set_type(std::string type);
    std::vector<chainset> get_chainsets(void);
    void add_chainset(chainset chainset);
private:
    int id;
    double util;
    std::string type;
    std::vector<chainset> chainsets;
    //std::vector<std::shared_ptr<chain>> chains;
};

struct timer_callback
{
    int timer_prio;
    double P;
    int timer_cpu;
};

class callback_row
{
public:
    double period;
    double cpu_time;
    double gpu_time;
    double tpu_time;
    double deadline;
    int chain_id;
    int order;
    int priority;
    int cpu_id;
    int executor_id;
    int bucket;
    callback_row(double, double, double, double, int, int, int, int, int, int, double);
};
class chainset
{
public:
    int num_executors = -1;
    int num_cpus = -1;
    int num_chains = -1;
    int num_callbacks = -1;
    std::vector<std::shared_ptr<cpu>> cpus;
    std::vector<std::shared_ptr<executor>> executors;
    std::vector<std::shared_ptr<callback>> callbacks;
    std::vector<std::shared_ptr<chain>> chains;
    chainset(std::vector<callback_row> data, int, int);
    std::vector<double> response_times;
    std::vector<double> deadlines;
    bool schedulable(void);
    void request_driven_gpu_bound(void);
    double job_driven_gpu_bound(std::shared_ptr<callback> t_callback, std::vector<std::shared_ptr<callback>> chain_callbacks, double R);
    timer_callback find_timer_callback(std::vector<std::shared_ptr<executor>> chain_executors, int chain_id);
    std::vector<callback_row> to_callback_row(void);
    std::vector<std::vector<callback_row>> to_callback_row_vector(void);
    std::vector<std::shared_ptr<accelerator>> accelerators;
};

class callback
{
public:
    int id;
    std::string type;
    double T;
    double C;
    double G;
    double epsilon;
    double D;
    double tpu_C;
    int priority;
    int cpu_id;
    int bucket;
    int chain_id;
    int chain_order;
    double chain_T;
    double chain_c;
    double wcrt;
    int executor;
    bool segment_flag;
    int segment_C;
    int segment_G;
    int segment_Tpu;
    bool chain_on_cpu;
    double gpu_waiting;
    double tpu_waiting;
    double gpu_handling;
    double segment_gpu_handling;
    double segment_tpu_handling;
    double segment_n_callbacks;
    int gpu_id = -1;
    int tpu_id = -1;
    callback(int id, int period, double execution, double gpu_execution, int chain_id, int order, int callback_prio, int cpu_id, int executor_id, int bucket, double tpu_C);
    callback(const callback &second);
    bool operator==(const callback &c);
    callback operator=(const callback &other);
    void swap(callback &first, callback &second);
    callback_row to_row(void);
};
class chain
{
public:
    int id;
    std::string type;
    int num_callbacks;
    std::vector<std::shared_ptr<callback>> t_callback;
    std::vector<std::shared_ptr<callback>> r_callbacks;
    //std::vector<std::shared_ptr<accelerator>> accelerators;
    int C;
    int T;
    int sem_priority;
    bool gpu;
    bool tpu;
    chain(int id, int sem_prio);
    void add_callback(std::shared_ptr<callback>);
    chain(const chain &second);
    chain operator=(const chain &other);
    void swap(chain &first, chain &second);
};

class cpu
{
public:
    int id;
    double utilization;
    std::vector<int> executor_ids;
    std::vector<std::shared_ptr<executor>> executors;
    cpu(int id);
    void assign_executor(std::shared_ptr<executor> exe);
    cpu(const cpu &second);
    cpu operator=(const cpu &other);
    void swap(cpu &first, cpu &second);
};

class executor
{
public:
    int id;
    std::string type;
    int priority;
    int cpu_id;
    std::vector<std::shared_ptr<callback>> callbacks;
    double util;
    executor(int id);
    void add_callback(std::shared_ptr<callback>);
    void assign(std::shared_ptr<callback>);
    executor(const executor &second);
    executor operator=(const executor &other);
    void swap(executor &first, executor &second);
};