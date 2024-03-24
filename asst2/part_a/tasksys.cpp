#include "tasksys.h"
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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
  // You do not need to implement this method.
  return 0;
}

void TaskSystemSerial::sync() {
  // You do not need to implement this method.
  return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char *TaskSystemParallelSpawn::name() {
  return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
    : ITaskSystem(num_threads), num_threads(num_threads) {
  //
  // TODO: CS149 student implementations may decide to perform setup
  // operations (such as thread pool construction) here.
  // Implementations are free to add new class member variables
  // (requiring changes to tasksys.h).
  //
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable *runnable, int num_total_tasks) {
  std::queue<std::function<void(void)>> task_queue;
  std::mutex task_mutex;
  std::vector<std::thread> thread_vector;
  for (int i = 0; i < num_total_tasks; i++) {
    task_queue.push([&runnable, i, num_total_tasks]() {
      runnable->runTask(i, num_total_tasks);
    });
  }
  for (int i = 0; i < num_threads; i++) {
    thread_vector.push_back(std::thread([&task_queue, &task_mutex]() {
      while (1) {
        std::unique_lock<std::mutex> lk(task_mutex);
        if (task_queue.empty())
          break;
        std::function<void(void)> task = std::move(task_queue.front());
        task_queue.pop();
        lk.unlock();
        task();
      }
    }));
  }
  for (int i = 0; i < num_threads; i++) {
    thread_vector[i].join();
  }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // You do not need to implement this method.
  return 0;
}

void TaskSystemParallelSpawn::sync() {
  // You do not need to implement this method.
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
    : ITaskSystem(num_threads), finished(false), remain_tasks(0),
      num_threads(num_threads) {
  for (int i = 0; i < num_threads; i++) {
    thread_vector.push_back(std::thread([&]() {
      while (!finished) {
        int cur_task_id;
        {
          std::unique_lock<std::mutex> lk(this->task_mutex);
          if (this->remain_tasks == 0)
            continue;
          cur_task_id = this->remain_tasks - 1;
          this->remain_tasks--;
        }
        runnable->runTask(cur_task_id, this->num_total_tasks);
        this->unfinished_tasks--;
      }
    }));
  }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
  finished = true;
  for (auto &thread : this->thread_vector) {
    thread.join();
  }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable *runnable,
                                               int num_total_tasks) {
  {
    std::unique_lock<std::mutex> lk(this->task_mutex);
    this->remain_tasks = num_total_tasks;
    this->num_total_tasks = num_total_tasks;
    this->unfinished_tasks = num_total_tasks;
    this->runnable = runnable;
  }
  while (this->unfinished_tasks) {
  }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {
  // You do not need to implement this method.
  return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
  // You do not need to implement this method.
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
    : ITaskSystem(num_threads), finished(false), remain_tasks(0),
      num_threads(num_threads) {

  for (int i = 0; i < num_threads; i++) {
    thread_vector.emplace_back(std::thread([&]() {
      while (1) {
        int cur_task_id;
        {
          std::unique_lock<std::mutex> lk(task_mutex);
          while (remain_tasks == 0 && !finished) {
            consumer.wait(lk);
          }
          if (finished)
            return;
          cur_task_id = remain_tasks - 1;
          remain_tasks--;
        }
        runnable->runTask(cur_task_id, num_total_tasks);

        std::unique_lock<std::mutex> lk(task_mutex);
        unfinished_tasks--;
        if (unfinished_tasks == 0) {
          producer.notify_one();
        }
      }
    }));
  }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
  {
    std::unique_lock<std::mutex> lk(task_mutex);
    finished = true;
  }
  consumer.notify_all();
  for (auto &thread : thread_vector) {
    thread.join();
  }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable,
                                               int num_total_tasks) {
  {
    std::unique_lock<std::mutex> lk(task_mutex);
    remain_tasks = num_total_tasks;
    this->num_total_tasks = num_total_tasks;
    this->unfinished_tasks = num_total_tasks;
    this->runnable = runnable;
  }
  consumer.notify_all();
  std::unique_lock<std::mutex> lk(task_mutex);
  while (unfinished_tasks > 0) {
    producer.wait(lk);
  }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(
    IRunnable *runnable, int num_total_tasks, const std::vector<TaskID> &deps) {

  //
  // TODO: CS149 students will implement this method in Part B.
  //

  return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

  //
  // TODO: CS149 students will modify the implementation of this method in Part
  // B.
  //

  return;
}
