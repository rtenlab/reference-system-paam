#include "admissions.h"
accelerator::accelerator(void)
{
    this->id = 0;
    this->util = 0;
    this->type = "none";
}
accelerator::accelerator(int id, double util, std::string type)
{
    this->id = id;
    this->util = util;
    this->type = type;
}
accelerator::accelerator(const accelerator &second)
{
    this->id = second.id;
    this->util = second.util;
    this->type = second.type;
}
accelerator accelerator::operator=(const accelerator &other)
{
    accelerator temp(other);
    swap(*this, temp);
    return *this;
}
void accelerator::swap(accelerator &first, accelerator &second)
{
    std::swap(first.id, second.id);
    std::swap(first.util, second.util);
    std::swap(first.type, second.type);
}
void accelerator::set_util(double util)
{
    this->util = util;
}
void accelerator::set_id(int id)
{
    this->id = id;
}
int accelerator::get_id(void)
{
    return this->id;
}
double accelerator::get_util(void)
{
    return this->util;
}

std::string accelerator::get_type(void)
{
    return this->type;
}
void accelerator::set_type(std::string type)
{
    this->type = type;
}
std::vector<chainset> accelerator::get_chainsets(void)
{
    return this->chainsets;
}
void accelerator::add_chainset(chainset chainset)
{
    chainset.id = this->chainsets.size();
    this->chainsets.push_back(chainset);
}
void accelerator::remove_chainset(chainset chainset)
{
    for (int i = 0; i < this->chainsets.size(); i++)
    {
        if (this->chainsets[i].id == chainset.id)
        {
            this->chainsets.erase(this->chainsets.begin() + i);
        }
    }
}

callback::callback(int id, int period, double execution, double gpu_execution, int chain_id, int order, int callback_prio, int cpu_id, int executor_id, int bucket, double tpu_C)
{
    this->id = id;
    this->T = period;
    this->tpu_C = tpu_C;
    if (period != 0)
    {
        this->type = "timer";
        this->D = period;
    }
    else
    {
        this->type = "regular";
    }
    this->C = execution;
    this->G = gpu_execution;
    this->epsilon = 0; // miscellaneous gpu execution (including overhead by AAMF)
    this->priority = callback_prio;
    this->chain_id = chain_id;
    this->chain_order = order;
    this->chain_c = 0;
    this->wcrt = 0;
    this->segment_flag = false;
    this->segment_C = 0;
    this->segment_G = 0;
    this->segment_gpu_handling = 0;
    this->segment_n_callbacks = 0;
    this->chain_on_cpu = false;
    this->bucket = bucket;
    this->executor = executor_id;
    this->cpu_id = cpu_id;
}
callback::callback(const callback &second)
{
    this->id = second.id;
    this->type = second.type;
    this->T = second.T;
    this->C = second.C;
    this->G = second.G;
    this->tpu_C = second.tpu_C;
    this->epsilon = second.epsilon;
    this->D = second.D;
    this->priority = second.priority;
    this->cpu_id = second.cpu_id;
    this->bucket = second.bucket;
    this->chain_id = second.chain_id;
    this->chain_order = second.chain_order;
    this->chain_c = second.chain_c;
    this->chain_T = second.chain_T;
    this->wcrt = second.wcrt;
    this->executor = second.executor;
    this->segment_flag = second.segment_flag;
    this->segment_C = second.segment_C;
    this->segment_G = second.segment_G;
    this->chain_on_cpu = second.chain_on_cpu;
    this->gpu_waiting = second.gpu_waiting;
    this->gpu_handling = second.gpu_handling;
    this->segment_gpu_handling = second.segment_gpu_handling;
    this->segment_n_callbacks = second.segment_n_callbacks;
    this->tpu_id = second.tpu_id;
    this->gpu_id = second.gpu_id;
}
bool callback::operator==(const callback &c)
{
    if (id == c.id && priority == c.priority && executor == c.executor && chain_id == c.chain_id && chain_order == c.chain_order && cpu_id == c.cpu_id && bucket == c.bucket)
        return true;
    return false;
}

callback callback::operator=(const callback &other)
{
    callback temp(other);
    swap(*this, temp);
    return *this;
}
chain::chain(const chain &second)
{
    this->id = second.id;
    this->type = second.type;
    this->num_callbacks = second.num_callbacks;
    this->t_callback = second.t_callback;
    this->r_callbacks = second.r_callbacks;
    this->C = second.C;
    this->T = second.T;
    this->sem_priority = second.sem_priority;
    this->gpu = second.gpu;
    this->tpu = second.tpu;
}
chain chain::operator=(const chain &other)
{
    chain temp(other);
    swap(*this, temp);
    return *this;
}
void chain::swap(chain &first, chain &second) // nothrow
{
    // the two objects are effectively swapped
    std::swap(first.id, second.id);
    std::swap(first.type, second.type);
    std::swap(first.num_callbacks, second.num_callbacks);
    std::swap(first.t_callback, second.t_callback);
    std::swap(first.r_callbacks, second.r_callbacks);
    std::swap(first.C, second.C);
    std::swap(first.T, second.T);
    std::swap(first.sem_priority, second.sem_priority);
    std::swap(first.gpu, second.gpu);
    std::swap(first.tpu, second.tpu);
}
void callback::swap(callback &first, callback &second) // nothrow
{
    // the two objects are effectively swapped
    std::swap(first.id, second.id);
    std::swap(first.type, second.type);
    std::swap(first.T, second.T);
    std::swap(first.C, second.C);
    std::swap(first.G, second.G);
    std::swap(first.tpu_C, second.tpu_C);
    std::swap(first.epsilon, second.epsilon);
    std::swap(first.D, second.D);
    std::swap(first.priority, second.priority);
    std::swap(first.cpu_id, second.cpu_id);
    std::swap(first.bucket, second.bucket);
    std::swap(first.chain_id, second.chain_id);
    std::swap(first.chain_order, second.chain_order);
    std::swap(first.chain_c, second.chain_c);
    std::swap(first.chain_T, second.chain_T);
    std::swap(first.wcrt, second.wcrt);
    std::swap(first.executor, second.executor);
    std::swap(first.segment_flag, second.segment_flag);
    std::swap(first.segment_C, second.segment_C);
    std::swap(first.segment_G, second.segment_G);
    std::swap(first.chain_on_cpu, second.chain_on_cpu);
    std::swap(first.gpu_waiting, second.gpu_waiting);
    std::swap(first.gpu_handling, second.gpu_handling);
    std::swap(first.segment_gpu_handling, second.segment_gpu_handling);
    std::swap(first.segment_n_callbacks, second.segment_n_callbacks);
    std::swap(first.tpu_id, second.tpu_id);
    std::swap(first.gpu_id, second.gpu_id);
}

cpu::cpu(const cpu &second)
{
    this->id = second.id;
    this->utilization = second.utilization;
    this->executors = second.executors;
    this->executor_ids = second.executor_ids;
}
cpu cpu::operator=(const cpu &other)
{
    cpu temp(other);
    swap(*this, temp);
    return *this;
}
void cpu::swap(cpu &first, cpu &second) // nothrow
{
    // the two objects are effectively swapped
    std::swap(first.id, second.id);
    std::swap(first.utilization, second.utilization);
    std::swap(first.executors, second.executors);
    std::swap(first.executor_ids, second.executor_ids);
}

executor::executor(const executor &second)
{
    this->id = second.id;
    this->type = second.type;
    this->priority = second.priority;
    this->callbacks = second.callbacks;
    this->cpu_id = second.cpu_id;
    this->util = second.util;
}
executor executor::operator=(const executor &other)
{
    executor temp(other);
    swap(*this, temp);
    return *this;
}
void executor::swap(executor &first, executor &second) // nothrow
{
    // the two objects are effectively swapped
    std::swap(first.id, second.id);
    std::swap(first.type, second.type);
    std::swap(first.priority, second.priority);
    std::swap(first.callbacks, second.callbacks);
    std::swap(first.cpu_id, second.cpu_id);
    std::swap(first.util, second.util);
}

std::vector<callback_row> chainset::to_callback_row(void)
{
    std::vector<callback_row> data;
    for (auto &callback : this->callbacks)
    {
        data.push_back(callback->to_row());
    }
    return data;
}

std::vector<std::vector<callback_row>> chainset::to_callback_row_vector(void)
{
    std::vector<std::vector<callback_row>> data;
    int num_chains = 0;
    for (auto &callback : this->callbacks)
    {
        if (callback->chain_id > num_chains)
        {
            num_chains = callback->chain_id;
        }
    }
    for (int i = 0; i < num_chains; i++)
    {
        std::vector<callback_row> temp;

        for (auto &callback : this->callbacks)
        {
            if (callback->chain_id == i)
            {
                temp.push_back(callback->to_row());
            }
        }
        data.push_back(temp);
        temp.clear();
    }

    return data;
}

callback_row callback::to_row(void)
{
    struct callback_row row(this->T, this->C, this->G, this->D, this->chain_id, this->chain_order, this->priority, this->cpu_id, this->executor, this->bucket, this->tpu_C);
    return row;
}

chainset::chainset(std::vector<callback_row> data, int num_gpus, int num_tpus)
{
    this->num_callbacks = data.size();
    int c = 0;
    int chain_idx = 0;

    for (int i = 0; i < num_gpus; i++)
    {
        this->accelerators.push_back(std::make_shared<accelerator>(accelerator(i, 0.0, "gpu")));
    }
    for (int i = 0; i < num_tpus; i++)
    {
        this->accelerators.push_back(std::make_shared<accelerator>(accelerator(i, 0.0, "tpu")));
    }

    for (auto &row : data)
    {
        this->num_cpus = row.cpu_id > this->num_cpus ? row.cpu_id : this->num_cpus;
        this->num_executors = row.executor_id > this->num_executors ? row.executor_id : this->num_executors;
        this->num_chains = row.chain_id > this->num_chains ? row.chain_id : this->num_chains;
    }

    for (int i = 0; i <= num_cpus; i++)
    {
        this->cpus.push_back(std::make_shared<cpu>(cpu(i)));
    }
    for (int i = 0; i <= num_executors; i++)
    {
        this->executors.push_back(std::make_shared<executor>(executor(i)));
    }
    for (auto &row : data)
    {
        auto temp_callback = std::make_shared<callback>(callback(c, row.period, row.cpu_time, row.gpu_time, row.chain_id, row.order, row.priority, row.cpu_id, row.executor_id, row.bucket, row.tpu_time));
        c++;
        temp_callback->chain_on_cpu = true;
        if (this->chains.size() <= row.chain_id && row.order == 1)
        {
            chain_idx = this->chains.size();
            auto temp_chain = std::make_shared<chain>(chain(chain_idx, row.priority));
            if (row.gpu_time > 0.0)
            {
                temp_chain->gpu = true;
            }
            if (row.tpu_time > 0.0)
            {
                temp_chain->tpu = true;
            }
            this->chains.push_back(temp_chain);
        }
        temp_callback->chain_id = chain_idx;
        this->callbacks.push_back(temp_callback);
        this->chains[chain_idx]->add_callback(temp_callback);
    }

    for (auto &chain : this->chains) // For each chain
    {
        double gpu_C = 0.0;
        double tpu_C = 0.0;
        for (auto &callback : chain->t_callback) // for each timer callback
        {
            gpu_C += callback->G;
            tpu_C += callback->tpu_C;
            this->executors.at(callback->executor)->add_callback(callback); // add the callback to the executor
        }
        for (auto &callback : chain->r_callbacks) // for each regular callback
        {
            gpu_C += callback->G;
            tpu_C += callback->tpu_C;
            this->executors.at(callback->executor)->add_callback(callback); // add the callback to the executor
        }
        int gpu_idx = 0;
        int tpu_idx = 0;
        std::shared_ptr<accelerator> gpu = nullptr;
        std::shared_ptr<accelerator> tpu = nullptr;
        // For each accelerator
        for (auto &acc : this->accelerators)
        {
            if (chain->gpu && !acc->get_type().compare("gpu") && acc->get_id() == gpu_idx)
            { // if the chain is gpu bound and the accelerator is a gpu and if the accelerator is the current gpu
                gpu = acc;
            }
            else if (chain->tpu && !acc->get_type().compare("tpu") && acc->get_id() == tpu_idx)
            { // else if the chain is tpu bound and the accelerator is a tpu and if the accelerator is the current tpu
                tpu = acc;
            }
        }
        // worst fit decreasing initial accelerator assignment
        for (auto &acc : this->accelerators)
        {
            if (chain->gpu)
            { // If the chain is gpu bound
                if (!acc->get_type().compare("gpu"))
                { // If the accelerator is a gpu
                    if (acc->get_util() < gpu->get_util())
                    {                            // If the accelerator is less utilized than the current gpu
                        gpu_idx = acc->get_id(); // Set the gpu_idx to the current gpu
                        gpu = acc;               // Set the gpu to the current gpu
                    }
                }
            }
            if (chain->tpu)
            { // If the chain is tpu bound
                if (!acc->get_type().compare("tpu"))
                { // If the accelerator is a tpu
                    if (acc->get_util() < tpu->get_util())
                    {                            // If the accelerator is less utilized than the current tpu
                        tpu_idx = acc->get_id(); // Set the tpu_idx to the current tpu
                        tpu = acc;               // Set the tpu to the current tpu
                    }
                }
            }
        }
        if (gpu != nullptr)
        {
            gpu->set_util(gpu->get_util() + gpu_C);
            printf("Chain %d assigned to GPU %d\n", chain->id, gpu->get_id());
        }
        else
        {
            printf("GPU IS NULL\n");
        }
        if (tpu != nullptr)
        {
            tpu->set_util(tpu->get_util() + tpu_C);
            printf("Chain %d assigned to TPU %d\n", chain->id, tpu->get_id());
        }
        else
        {
            printf("TPU IS NULL\n");
        }
        int gid = -1, tid = -1;
        if (gpu != nullptr)
        {
            gid = gpu->get_id();
        }
        if (tpu != nullptr)
        {
            tid = tpu->get_id();
        }
        chain->assign_accelerators_to_callbacks(gid, tid);
    }
    for (auto &e : this->executors)
    {
        this->cpus.at(e->cpu_id)->assign_executor(e);
    }
}
bool chainset::schedule_multiple_accelerators(void) 
// REWORK so that we do not reassign 
// existing chains to different accelerators.
{
    bool val = this->schedulable();
    if (val)
    {
        return val;
    }
    else
    {
        if (num_gpus > 0)
        {
            if (num_tpus > 0)
            {
                // For all gpus and tpus in the system, 
                // iterate through all different combinations 
                // and invoke the schedulability test until we have a successful test
            }
        }
    }
}
void chain::assign_accelerators_to_callbacks(int gpu_id, int tpu_id)
{
    if (tpu_id != -1)
    {
        for (auto &callback : this->t_callback)
        {
            callback->tpu_id = tpu_id;
        }
        for (auto &callback : this->r_callbacks)
        {
            callback->tpu_id = tpu_id;
        }
    }
    if (gpu_id != -1)
    {
        for (auto &callback : this->t_callback)
        {
            callback->gpu_id = gpu_id;
        }
        for (auto &callback : this->r_callbacks)
        {
            callback->gpu_id = gpu_id;
        }
    }
}
double chainset::schedulable(void)
{
    this->request_driven_gpu_bound();
    this->request_driven_tpu_bound();
    std::vector<std::shared_ptr<callback>> sch_callbacks;
    std::vector<std::shared_ptr<executor>> sch_executors;

    int idx = 0;
    for (auto &c : this->cpus)
    {
        std::vector<int> chain_segment_priority(this->chains.size(), 100);
        std::vector<int> chain_segment_task_idx(this->chains.size(), 0);
        std::vector<double> chain_segment_exe_time(this->chains.size(), 0);
        std::vector<double> chain_segment_gpu_time(this->chains.size(), 0);
        std::vector<double> chain_segment_gpu_handling(this->chains.size(), 0);
        std::vector<int> chain_segment_n_callbacks(this->chains.size(), 0);
        std::vector<double> chain_segment_tpu_time(this->chains.size(), 0);
        std::vector<double> chain_segment_tpu_handling(this->chains.size(), 0);
        for (auto &e : c->executors)
        {
            for (auto &t : e->callbacks)
            {
                int curr_chain = t->chain_id;
                if (t->priority < chain_segment_priority[curr_chain])
                {
                    chain_segment_priority[curr_chain] = t->priority;
                    chain_segment_task_idx[curr_chain] = idx;
                }
                chain_segment_exe_time[curr_chain] += t->C;
                chain_segment_gpu_time[curr_chain] += t->G;
                chain_segment_tpu_time[curr_chain] += t->tpu_C;
                chain_segment_gpu_handling[curr_chain] += t->gpu_handling;
                chain_segment_n_callbacks[curr_chain] += 1;
                sch_callbacks.push_back(t);
                idx++;
            }
            sch_executors.push_back(e);
        }
        // Set swgment callbacks
        for (int i = 0; i < this->chains.size(); i++)
        {
            if (chain_segment_exe_time[i] != 0)
            {
                sch_callbacks[chain_segment_task_idx[i]]->segment_flag = true;
                sch_callbacks[chain_segment_task_idx[i]]->segment_C = chain_segment_exe_time[i];
                sch_callbacks[chain_segment_task_idx[i]]->segment_G = chain_segment_gpu_time[i];
                sch_callbacks[chain_segment_task_idx[i]]->segment_Tpu = chain_segment_tpu_time[i];
                sch_callbacks[chain_segment_task_idx[i]]->segment_tpu_handling = chain_segment_tpu_handling[i];
                sch_callbacks[chain_segment_task_idx[i]]->segment_gpu_handling = chain_segment_gpu_handling[i];
                sch_callbacks[chain_segment_task_idx[i]]->segment_n_callbacks = chain_segment_n_callbacks[i];
            }
        }
    }
    for (auto &s : this->chains)
    {
        s->t_callback.clear();
        s->r_callbacks.clear();
    }
    std::sort(sch_executors.begin(), sch_executors.end(), [](const auto &lhs, const auto &rhs)
              { return lhs->priority > rhs->priority; });
    // Sort callbacks in descending priority order
    std::sort(sch_callbacks.begin(), sch_callbacks.end(), [](const auto &lhs, const auto &rhs)
              { return lhs->priority > rhs->priority; });

    for (auto &callback : sch_callbacks)
    {
        bool flag = true;
        int t_id = callback->id;
        bool segment_flag = callback->segment_flag;
        int t_exe = callback->executor;
        int t_prio = callback->priority;
        int t_chain = callback->chain_id;
        bool t_chain_cpu = callback->chain_on_cpu;
        int t_cpu = callback->cpu_id;
        int t_bucket = callback->bucket;
        int t_gpu = callback->gpu_id;
        int t_tpu = callback->tpu_id;

        double B = 0;

        for (auto &c : sch_executors[t_exe]->callbacks)
        {
            if (c->chain_id != t_chain && c->priority < t_prio && c->C > B)
            {
                B = c->C;
            }
        }
        if (!segment_flag)
        {
            continue;
        }
        double R = callback->segment_C + B;
        double R_prev = R;
        double P = 0.0;
        while (flag)
        {
            double W = 0;
            for (auto &j : sch_callbacks) // for all callbacks
            {
                if (j->segment_flag == false) // if not a segment
                {
                    continue;
                }
                if (j->id != t_id && sch_executors[t_exe]->priority != executors[j->executor]->priority) // if not the same callback and not the same executor priority
                {
                    if (((t_chain_cpu && j->chain_id != t_chain) && j->cpu_id == t_cpu) || ((!t_chain_cpu) && j->cpu_id == t_cpu)) // if not the same chain and same cpu
                    {
                        timer_callback tc = find_timer_callback(sch_executors, j->chain_id);
                        if (j->chain_on_cpu)
                        {
                            P = std::max(j->chain_c, tc.P);
                        }
                        else
                        {
                            P = tc.P;
                        }
                        if (sch_executors[t_exe]->priority == sch_executors[j->executor]->priority && t_prio < j->priority)
                        {
                            W += std::ceil(R / P) * (j->segment_C + j->segment_gpu_handling);
                        }
                        if (sch_executors[t_exe]->priority < sch_executors[j->executor]->priority)
                        {
                            if (tc.timer_prio >= t_prio || tc.timer_cpu != t_cpu)
                            {
                                W += std::ceil(R / P) * (j->C);
                            }
                            else
                            {
                                W += j->C;
                            }
                        }
                    }
                }
            }

            double gpu_handling = job_driven_gpu_bound(callback, sch_callbacks, R);
            if (gpu_handling < callback->segment_gpu_handling)
            {
                callback->segment_gpu_handling = gpu_handling;
            }
            double tpu_handling = job_driven_tpu_bound(callback, sch_callbacks, R);
            if (tpu_handling < callback->segment_tpu_handling)
            {
                callback->segment_tpu_handling = tpu_handling;
            }

            R = W = callback->segment_C + callback->segment_gpu_handling + callback->segment_tpu_handling + B;
            if (R <= R_prev)
            {
                callback->wcrt = R;
                break;
            }
            R_prev = R;
        }
    }

    for (auto &c : sch_callbacks)
    {
        if (!c->type.compare("timer"))
        {
            this->chains[c->chain_id]->t_callback.push_back(c);
        }
        else
        {
            this->chains[c->chain_id]->r_callbacks.push_back(c);
        }
    }

    std::vector<double> chain_latency(this->chains.size(), 0);
    int z = 0;
    for (auto chain : this->chains)
    {
        for (auto &t : chain->t_callback)
        {
            if (t->segment_flag)
            {
                chain_latency[z] += t->wcrt;
            }
        }
        for (auto &t : chain->r_callbacks)
        {
            if (t->segment_flag)
            {
                chain_latency[z] += t->wcrt;
            }
        }
        for (auto &t : chain->t_callback)
        {
            if (chain_latency[z] > t->T)
            {
                chain_latency[z] += t->T;
            }
        }
        z++;
    }
    z = 0;
    double chains_schedulable = 0;

    for (auto &c : this->chains)
    {
        if (c->T >= chain_latency[z])
        {   chains_schedulable += 1;
            //return false;
        }
        z++;
    }
    return chains_schedulable / this->chains.size();
    //return true;
}
void chainset::request_driven_gpu_bound(void)
{
    // For each callback
    for (auto &callback : this->callbacks)
    {
        double max_gpu_exe = 0;
        // Find the maximum GPU execution time of all other callbacks on the same gpu with higher priority
        for (auto &other_callback : this->callbacks)
        { // If the other callback has higher priority and is not in the same chain and is in the same bucket on the same gpu
            if (other_callback->priority < callback->priority && other_callback->chain_id != callback->chain_id && other_callback->bucket == callback->bucket && callback->gpu_id == other_callback->gpu_id && other_callback->G + other_callback->epsilon > max_gpu_exe)
            {
                // Set the maximum GPU execution time to the other callback's GPU execution time
                max_gpu_exe = other_callback->G + other_callback->epsilon;
            }
        }
        double beta = max_gpu_exe;
        double beta_prev = 0;
        while (true)
        {
            double temp = 0;
            // For each callback
            for (auto &selected_callback : this->callbacks)
            {
                // If the selected callback has higher chain priority and is not the same callback and is not in the same chain
                if (this->chains.at(selected_callback->chain_id)->sem_priority > this->chains.at(callback->chain_id)->sem_priority && selected_callback->id != callback->id && selected_callback->chain_id != callback->chain_id && selected_callback->gpu_id == callback->gpu_id)
                {
                    // Add the ceiling of the beta value divided by the period of the chain plus 1 multiplied by the sum of the GPU execution time and epsilon of the selected callback to the temp variable
                    temp += (ceil(beta / this->chains.at(selected_callback->chain_id)->T) + 1) * (selected_callback->G + selected_callback->epsilon);
                }
            }
            // Set the beta value to the maximum GPU execution time plus the temp variable
            beta = max_gpu_exe + temp;
            if (abs(beta - beta_prev) < 0.001 || beta > 10000000)
            {
                callback->gpu_waiting = beta;
                callback->gpu_handling = beta + callback->G + 2 * callback->epsilon;
                break;
            }
            beta_prev = beta;
        }
    }
}

double chainset::job_driven_gpu_bound(std::shared_ptr<callback> t_callback, std::vector<std::shared_ptr<callback>> chain_callbacks, double R)
{
    double max_gpu_exe = 0;
    double beta = 0;
    // For all callbacks in the current chainset
    for (auto &callback : chain_callbacks)
    {
        // If the callback is higher priority and not in the same chain and in the same bucket on the same gpu and the GPU execution time plus epsilon is greater than the maximum GPU execution time
        if (callback->priority < t_callback->priority && callback->chain_id != t_callback->chain_id &&
            callback->bucket == t_callback->bucket && callback->gpu_id == t_callback->gpu_id && callback->G + callback->epsilon > max_gpu_exe)
        {
            // set maximum GPU execution time to the GPU execution time plus epsilon
            max_gpu_exe = callback->G + callback->epsilon;
        }
    }
    beta = max_gpu_exe * t_callback->segment_n_callbacks;

    // For all callbacks in the current chainset
    for (auto &callback : chain_callbacks)
    {
        // If the callback is higher chain priority and not the same callback and not in the same chain
        if (this->chains.at(callback->chain_id)->sem_priority > this->chains.at(t_callback->chain_id)->sem_priority &&
            callback->id != t_callback->id && callback->chain_id != t_callback->chain_id && callback->gpu_id == t_callback->gpu_id)
        {
            beta = beta + (ceil(R / this->chains.at(callback->chain_id)->T) + 1) * (callback->G + callback->epsilon);
        }
    }
    return beta + t_callback->segment_G + 2 * t_callback->epsilon;
}

void chainset::request_driven_tpu_bound(void)
{
    // For each callback
    for (auto &callback : this->callbacks)
    {
        double max_tpu_exe = 0;
        // Find the maximum GPU execution time of all other callbacks on the same gpu with higher priority
        for (auto &other_callback : this->callbacks)
        { // If the other callback has higher priority and is not in the same chain and is in the same bucket on the same gpu
            if (other_callback->priority < callback->priority && other_callback->chain_id != callback->chain_id && callback->tpu_id == other_callback->tpu_id && other_callback->tpu_C + other_callback->epsilon > max_tpu_exe)
            {
                // Set the maximum GPU execution time to the other callback's GPU execution time
                max_tpu_exe = other_callback->tpu_C + other_callback->epsilon;
            }
        }
        double beta = max_tpu_exe;
        double beta_prev = 0;
        while (true)
        {
            double temp = 0;
            // For each callback
            for (auto &selected_callback : this->callbacks)
            {
                // If the selected callback has higher chain priority and is not the same callback and is not in the same chain
                if (this->chains.at(selected_callback->chain_id)->sem_priority > this->chains.at(callback->chain_id)->sem_priority && selected_callback->id != callback->id && selected_callback->chain_id != callback->chain_id && selected_callback->tpu_id == callback->tpu_id)
                {
                    // Add the ceiling of the beta value divided by the period of the chain plus 1 multiplied by the sum of the GPU execution time and epsilon of the selected callback to the temp variable
                    temp += (ceil(beta / this->chains.at(selected_callback->chain_id)->T) + 1) * (selected_callback->tpu_C + selected_callback->epsilon);
                }
            }
            // Set the beta value to the maximum GPU execution time plus the temp variable
            beta = max_tpu_exe + temp;
            if (abs(beta - beta_prev) < 0.001 || beta > 10000000)
            {
                callback->tpu_waiting = beta;
                callback->tpu_handling = beta + callback->tpu_C + 2 * callback->epsilon;
                break;
            }
            beta_prev = beta;
        }
    }
}

double chainset::job_driven_tpu_bound(std::shared_ptr<callback> t_callback, std::vector<std::shared_ptr<callback>> chain_callbacks, double R)
{
    double max_tpu_exe = 0;
    double beta = 0;
    // For all callbacks in the current chainset
    for (auto &callback : chain_callbacks)
    {
        // If the callback is higher priority and not in the same chain and in the same bucket on the same gpu and the GPU execution time plus epsilon is greater than the maximum GPU execution time
        if (callback->priority < t_callback->priority && callback->chain_id != t_callback->chain_id &&
            callback->tpu_id == t_callback->tpu_id && callback->tpu_C + callback->epsilon > max_tpu_exe)
        {
            // set maximum GPU execution time to the GPU execution time plus epsilon
            max_tpu_exe = callback->tpu_C + callback->epsilon;
        }
    }
    beta = max_tpu_exe * t_callback->segment_n_callbacks;

    // For all callbacks in the current chainset
    for (auto &callback : chain_callbacks)
    {
        // If the callback is higher chain priority and not the same callback and not in the same chain
        if (this->chains.at(callback->chain_id)->sem_priority > this->chains.at(t_callback->chain_id)->sem_priority &&
            callback->id != t_callback->id && callback->chain_id != t_callback->chain_id && callback->gpu_id == t_callback->gpu_id)
        {
            beta = beta + (ceil(R / this->chains.at(callback->chain_id)->T) + 1) * (callback->G + callback->epsilon);
        }
    }
    return beta + t_callback->segment_G + 2 * t_callback->epsilon;
}

timer_callback chainset::find_timer_callback(std::vector<std::shared_ptr<executor>> chain_executors, int chain_id)
{
    timer_callback tc;
    for (auto &executor : chain_executors)
    {
        for (auto &callback : executor->callbacks)
        {
            if (callback->chain_id == chain_id && !callback->type.compare("timer"))
            {
                tc.timer_prio = callback->priority;
                tc.P = callback->T;
                tc.timer_cpu = callback->cpu_id;
                return tc;
            }
        }
    }
    return tc;
}

void chain::add_callback(std::shared_ptr<callback> callback)
{
    // Check to see if callback is timer or regular
    if (callback->type.compare("timer") == 0)
    {
        this->t_callback.push_back(callback); // Update timer callback list
        this->T = callback->T;                // set timer callback period
    }
    else
    {
        this->r_callbacks.push_back(callback); // update regular callback list
    }
    // update the number of callbacks
    this->num_callbacks += 1;
    // Update C
    this->C += callback->C;
    // For each timer callback, update T and C
    for (auto c : this->t_callback)
    {
        c->chain_c = this->C;
        c->chain_T = this->T;
    }

    // For each regular callback, update T and C
    for (auto c : this->r_callbacks)
    {
        c->chain_c = this->C;
        c->chain_T = this->T;
    }

    if (this->sem_priority == 0 || this->sem_priority < callback->priority)
    {
        this->sem_priority = callback->priority;
    }
}

chain::chain(int id, int sem_prio)
{
    this->id = id;
    this->sem_priority = sem_prio;
    this->num_callbacks = 0;
    this->type = "p";
    this->C = 0;
    this->T = 0;
    if (this->sem_priority == 0 || this->sem_priority < sem_prio)
    {
        this->sem_priority = sem_prio;
    }
}

cpu::cpu(int id)
{
    this->id = id;
    this->utilization = 0;
}

void cpu::assign_executor(std::shared_ptr<executor> exe)
{
    exe->cpu_id = this->id;
    for (auto &t : exe->callbacks)
    {
        t->cpu_id = this->id;
    }
    this->executors.push_back(exe);
    this->executor_ids.push_back(exe->id);
    this->utilization += exe->util;
}

executor::executor(int id)
{
    this->id = id;
    this->priority = 0;
    this->type = "";
    this->util = 0;
    this->cpu_id = 0;
}

void executor::add_callback(std::shared_ptr<callback> task)
{
    int cpu_id = -1;
    this->util += task->C / task->chain_T;
    task->executor = this->id;
    this->callbacks.push_back(task);
    if (cpu_id == -1)
    {
        cpu_id = task->cpu_id;
    }
    else if (cpu_id != task->cpu_id)
    {
        printf("EXECUTOR ADD CALLBACK ERROR\n");
    }
    if (this->priority < task->priority)
    {
        this->priority = task->priority;
    }
    this->cpu_id = cpu_id;
}
void executor::assign(std::shared_ptr<callback> callback)
{
    this->callbacks.push_back(callback);
    callback->executor = this->id;
    callback->priority = this->priority;
    if (this->type.empty())
    {
        if (callback->chain_id != -1)
        {
            this->type = "chain";
        }
        else
        {
            this->type = "single";
        }
    }
}
callback_row::callback_row(double period, double cpu_time, double gpu_time, double deadline, int chain_id, int order, int priority, int cpu_id, int executor_id, int bucket, double tpu_time)
{
    this->period = period;
    this->cpu_time = cpu_time;
    this->gpu_time = gpu_time;
    this->tpu_time = tpu_time;
    this->deadline = deadline;
    this->chain_id = chain_id;
    this->order = order;
    this->priority = priority;
    this->cpu_id = cpu_id;
    this->executor_id = executor_id;
    this->bucket = bucket;
}


std::vector<double> rtas_variable_chains_test(void){
    int trails = 1000;
    int num_callbacks = 4;
    double cpu_util = 0.1;
    int num_chains_per_chainset = 12;
    std::vector<double> util_test;
    for (double i = 0.05 ; i <= 0.70; i+=0.05){
        util_test.push_back(i);
    }
    for ( auto util : util_test){
        for ( int num_chains = 1; num_chains <= num_chains_per_chainset; num_chains++){
            std::vector<double> trial_results;
            for (int trial = 0; trial < trials; trial++){
                std::vector<callback_row> test_config;
                for ( auto i = 0; i < num_chains; i++){
                    int period_rand = 100 + (rand() % 1000);
                    period_rand = period_rand - (period_rand % 10);
                    int prio = 99-i;
                    int bucket = prio/17;
                    for ( auto j = 0; j < num_callbacks; j++){
                        if ( j == 1){
                            callback_row chain_config(period_rand, period_rand*util/2/num_callbacks, period_rand*util/2/num_callbacks, period_rand, i, j, prio, 2+(i%6), i , bucket, 0);
                            test_config.push_back(chain_config);
                        }
                        else{
                            callback_row chain_config(0, period_rand*util/2/num_callbacks, period_rand*util/2/num_callbacks, 0, i, j, prio, 2+(i%6), i , bucket, 0);
                            test_config.push_back(chain_config);
                        }
                    }

                }
            }
    }
    return test_config_results;
}
std::vector<double> rtas_variable_util_ratio_test(void){
    int num_chains = 20;
    int num_callbacks;
    int fixed_chains = 4;
    int fixed_callbacks = 4;
    std::vector<double> utilization;
    double fixed_utilization = 0.05;
    double fixed_cpu_utilization = 0.001;
    int fixed_period = 250;

    std::vector<double> test_config_results;

    for ( auto i = 0; i < num_callbacks; i++){
        std::vector<callback_row> test_config;
        for ( auto j = 0; j < fixed_chains; j++){
            for( auto k = 0; k < i; k++){
                if( k == 1){
                    callback_row chain_config(fixed_period, fixed_period*fixed_utilization/2, fixed_period*fixed_utilization/2, fixed_period, j, k, 99 - j, 2+j, j , j, 0);
                    test_config.push_back(chain_config);

                }
                else{
                    callback_row chain_config(0, fixed_period*fixed_utilization/2, fixed_period*fixed_utilization/2, 0, j, k, 99 - j, 2+j, j , j, 0);
                    test_config.push_back(chain_config);

                }
            }
        }
        chainset test(test_config, 1, 1);
        test_config_results.push_back(test.schedulable());
    }
    return test_config_results;
}
int main(void)
{
    std::vector<callback_row> data;
    callback_row r1(220, 1, 2, 220, 0, 1, 98, 0, 0, 0, 10);
    callback_row r2(0, 1, 2, 0, 0, 2, 99, 0, 0, 0, 0);
    callback_row r3(220, 1, 2, 220, 1, 1, 96, 1, 1, 1, 0);
    callback_row r4(0, 1, 2, 0, 1, 2, 97, 1, 1, 1, 10);
    data.push_back(r1);
    data.push_back(r2);
    data.push_back(r3);
    data.push_back(r4);
    chainset test(data, 2, 1);
    if (test.schedulable())
    {
        printf("Chainset is Schedulable\n");
    }
    else
    {
        printf("Chainset is not Schedulable\n");
    }
    return 0;
}