#include "bfs.h"

#include <cstddef>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define VERBOSE
#define THRESHHOLD 10000000

void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                   int *distances) {
#pragma omp parallel 
{
    int local_num = 0;
    int *local_frontier = (int*)malloc(sizeof(int) * g->num_nodes);
    #pragma omp for schedule (dynamic, 200)
    for (int i = 0; i < frontier->count; i++) {
      int node = frontier->vertices[i];
      int start_edge = g->outgoing_starts[node];
      int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                                : g->outgoing_starts[node + 1];
  // attempt to add all neighbors to the new frontier
      for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int outgoing = g->outgoing_edges[neighbor];
        if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER,
                                        distances[node] + 1)) {
          local_frontier[local_num++] = outgoing;
        }
      }
    }
    int index = __sync_fetch_and_add(&new_frontier->count, local_num);
    memcpy(&new_frontier->vertices[index], local_frontier, local_num * sizeof(int));
    free(local_frontier);
  }
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}
void bottom_up_step(Graph g, vertex_set* frontier, vertex_set* new_frontier, int *distances, int iteration) {
#pragma omp parallel 
{
  int local_num = 0;
  int *local_frontier = (int*)malloc(sizeof(int) * g->num_nodes);
  #pragma omp for schedule(dynamic,200)
  for (int new_node = 0; new_node < g->num_nodes; new_node++) {
    if (distances[new_node] != NOT_VISITED_MARKER) continue;
    // 判断有没有入边的起点在frontier中
    int start_edge = g->incoming_starts[new_node];
    int end_edge = (new_node == g->num_nodes - 1) ? g->num_edges
                                            : g->incoming_starts[new_node + 1];
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int incoming = g->incoming_edges[neighbor];
      if (distances[incoming] == iteration) {
        distances[new_node] = iteration + 1;
        local_frontier[local_num++] = new_node;
        break;
      }
    }
  }
  int index = __sync_fetch_and_add(&new_frontier->count, local_num);
  memcpy(&new_frontier->vertices[index], local_frontier, sizeof(int) * local_num);
  free(local_frontier);
}
}

void bfs_bottom_up(Graph graph, solution *sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  int iteration = 0;
  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    bottom_up_step(graph, frontier, new_frontier, sol->distances, iteration++);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  // CS149 students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  int iteration = 0;
  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    if (frontier->count >= THRESHHOLD) {
      bottom_up_step(graph, frontier, new_frontier, sol->distances, iteration);
    } else {
      top_down_step(graph, frontier, new_frontier, sol->distances);
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
    iteration++;
  }
}
