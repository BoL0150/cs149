#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <queue>
#include <vector>
#include <thread>
#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

class Bulk {
    public:
        Bulk (TaskID bulk_id, std::queue<std::function<void(void)>> && q, const std::vector<TaskID> &deps) : bulk_id(bulk_id), task_queue(q) {
            for (auto id : deps) {
                this->deps.insert(id);
            }
        };
        TaskID bulk_id;
        std::queue<std::function<void(void)>> task_queue;
        std::unordered_set<TaskID>deps;
};
/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    private:
        std::atomic<int> task_id = 0;
        std::condition_variable producer;
        std::condition_variable consumer;
        std::vector<std::thread> thread_vector;
        std::mutex ready_q_mtx;
        std::queue<std::shared_ptr<Bulk>> ready_bulk_queue;
        // std::mutex waiting_q_mtx;
        // std::queue<std::function<void(void)>> waiting_task_queue;
        std::mutex wait_for_mtx;
        std::unordered_map<TaskID, std::vector<std::shared_ptr<Bulk>>> waits_for;

        int num_threads;
        std::atomic<int> remain_tasks;
        bool stop;
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

#endif
