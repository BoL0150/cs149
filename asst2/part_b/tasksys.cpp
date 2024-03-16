#include "tasksys.h"
#include <iostream>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char *TaskSystemSerial::name() { return "Serial"; }

TaskSystemSerial::TaskSystemSerial(int num_threads)
    : ITaskSystem(num_threads) {}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable *runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable *runnable,
                                          int num_total_tasks,
                                          const std::vector<TaskID> &deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() { return; }

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
    : ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable *runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(
    int num_threads)
    : ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable *runnable,
                                               int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement
    // TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(
    int num_threads)
    : ITaskSystem(num_threads), task_id(0), stop(false),
      num_threads(num_threads) {
    for (int i = 0; i < num_threads; i++) {
        thread_vector.emplace_back(std::thread([&]() {
            while (1) {
                std::shared_ptr<Bulk> cur_bulk;
                std::function<void(void)> cur_task;
                {
                    std::unique_lock<std::mutex> lk(ready_q_mtx);
                    while (ready_bulk_queue.empty() && !stop) {
                        consumer.wait(lk);
                    }
                    if (stop)
                        return;
                    cur_bulk = ready_bulk_queue.front();
                    cur_task = std::move(cur_bulk->task_queue.front());
                    cur_bulk->task_queue.pop();
                    // 如果bulk的任务队列空了，说明bulk执行完了，将它在ready队列中的条目删除
                    if (cur_bulk->task_queue.empty())
                        ready_bulk_queue.pop();
                }
                cur_task();
                cur_bulk->task_num--;
                // bulk的所有任务执行完了
                if (cur_bulk->task_num == 0) {
                    std::vector<std::shared_ptr<Bulk>> ready_bulks;
                    {
                        std::unique_lock<std::mutex> lk(wait_for_mtx);
                        std::vector<std::shared_ptr<Bulk>> &blocked_by_cur_bulk = waits_for[cur_bulk->bulk_id];
                        // 遍历所有被当前bulk阻塞的bulk
                        for (auto &blocked_bulk : blocked_by_cur_bulk) {
                            blocked_bulk->deps.erase(cur_bulk->bulk_id);
                            // 如果bulk的依赖全部消失了，那么该bulk就可以执行
                            if (blocked_bulk->deps.empty()) {
                                ready_bulks.push_back(std::move(blocked_bulk));
                            }
                        }
                        // bulk执行完了就要从waits_for中移除
                        waits_for.erase(cur_bulk->bulk_id);
                        if (waits_for.empty()) sync_cv.notify_one();
                    }
                    if (!ready_bulks.empty()) {
                        std::unique_lock<std::mutex> ready_lk(ready_q_mtx);
                        for (auto &bulk : ready_bulks) {
                            ready_bulk_queue.push(std::move(bulk));
                        }
                        consumer.notify_all();
                    }
                }
            }
        }));
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    {
        std::unique_lock<std::mutex> lk(ready_q_mtx);
        stop = true;
    }
    consumer.notify_all();
    // std::cout << "fuck you" << std::endl;
    for (auto &thread : thread_vector) {
        static int i = 0;
        thread.join();
        // std::cout << "finished join thread:" << i++ << std::endl;
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable,
                                               int num_total_tasks) {
    const std::vector<TaskID> deps;
    runAsyncWithDeps(runnable, num_total_tasks, deps);
    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {

    TaskID cur_id = task_id++;
    std::queue<std::function<void(void)>> task_queue;
    for (int i = 0; i < num_total_tasks; i++) {
        task_queue.emplace([runnable, i, num_total_tasks]() {
            runnable->runTask(i, num_total_tasks);
        });
    }
    {
        std::unique_lock<std::mutex> lk(wait_for_mtx);
        // 正在运行或者等待运行的bulk都要记录在waits_for中
        waits_for[cur_id] = std::vector<std::shared_ptr<Bulk>>();
    }
    bool is_ready = true;
    std::shared_ptr<Bulk> cur_bulk = std::make_shared<Bulk>(cur_id, std::move(task_queue));
    for (auto id : deps) {
        std::unique_lock<std::mutex> lk(wait_for_mtx);
        // 如果一个bulk的id在waits_for中不存在，就说明该bulk已经结束
        if (!waits_for.count(id))
            continue;
        // 如果没有结束，则记录依赖
        cur_bulk->deps.insert(id);
        is_ready = false;
        waits_for[id].push_back(cur_bulk);
    }

    if (deps.empty() || is_ready) {
        std::unique_lock<std::mutex> lk(ready_q_mtx);
        ready_bulk_queue.push(cur_bulk);
        consumer.notify_all();
    }
    return cur_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    std::unique_lock<std::mutex> lk(wait_for_mtx);
    while (!waits_for.empty()) {
        sync_cv.wait(lk);
    }
}
