#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads), num_threads(num_threads), remain_tasks(0), stop(false) {
    for (int i = 0; i < num_threads; i++) {
        thread_vector.emplace_back(std::thread([&]() {
            while (1) {
                Bulk *cur_bulk;
                std::function<void(void)> cur_task;
                {
                    std::unique_lock<std::mutex> lk(ready_q_mtx);
                    while (ready_bulk_queue.empty() && !stop) {
                        consumer.wait(lk);
                    }
                    if (stop) return;
                    cur_bulk = &ready_bulk_queue.front();
                    cur_task = std::move(cur_bulk->task_queue.front());
                    cur_bulk->task_queue.pop();
                }
                cur_task();
                {
                    std::unique_lock<std::mutex> lk(ready_q_mtx);
                    // 如果bulk的任务队列空了，说明bulk执行完了
                    if (cur_bulk->task_queue.empty()) {
                        ready_bulk_queue.pop();
                        std::vector<Bulk> &blocked_by_cur_bulk = waits_for[cur_bulk->bulk_id];
                        // 遍历所有被当前bulk阻塞的bulk
                        for (auto & blocked_bulk : blocked_by_cur_bulk) {
                            blocked_bulk.deps.erase(cur_bulk->bulk_id);
                            // 如果bulk的依赖全部消失了，那么该bulk就可以执行
                            if (blocked_bulk.deps.empty()) {
                                ready_bulk_queue.emplace(std::move(blocked_bulk));
                            }
                        }
                   }
                    
                        
                    }
                     
                }
            }
        }));
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {

    TaskID cur_id = task_id++;
    std::queue<std::function<void(void)>> task_queue;
    for (int i = 0; i < num_total_tasks; i++) {
        task_queue.emplace([&runnable, i, num_total_tasks] () {
            runnable->runTask(i, num_total_tasks);
        });
    }
    bool is_ready = true;
    std::shared_ptr<Bulk> cur_bulk = make_shared<Bulk>(cur_id, std::move(task_queue), deps);
    for (auto id : deps) {
        std::unique_lock<std::mutex> lk(wait_for_mtx);
        if (!waits_for.count(id)) continue;
        is_ready = false;
        waits_for[id].emplace_back(cur_id, std::move(task_queue), deps);
        break;
    }
    if (deps.empty() || is_ready) {
        std::unique_lock<std::mutex> lk (ready_q_mtx);
        ready_bulk_queue.emplace(cur_id, std::move(task_queue), deps);
        consumer.notify_all();
    } 
    waits_for[cur_id] = std::vector<Bulk>();
    return cur_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
