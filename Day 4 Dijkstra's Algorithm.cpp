#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <tuple>
#include <limits>

using namespace std;

class Solution {
public:
    /**
     * Implements Dijkstra's algorithm to find the shortest distance from a source vertex 
     * to all other vertices in a weighted, directed graph.
     * * @param n The number of vertices (0 to n-1).
     * @param edges A list of edges in the form (u, v, w).
     * @param src The source vertex.
     * @return An unordered_map where keys are vertices and values are their shortest 
     * distance from 'src'. Unreachable vertices have a distance of -1.
     */
    unordered_map<int, int> shortestPath(int n, vector<vector<int>>& edges, int src) {
        // --- 1. Build the Adjacency List (Graph Representation) ---
        // adj[u] will store a list of pairs {v, w} representing an edge u -> v with weight w
        unordered_map<int, vector<pair<int, int>>> adj;
        for (const auto& edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];
            adj[u].push_back({v, w});
        }

        // --- 2. Initialization ---
        
        // distance_map: Stores the shortest distance from 'src' to each vertex.
        // Initialize all distances to a large value (infinity) to represent unreachability.
        // We use an unordered_map to store distances only for vertices involved,
        // though an array/vector is often used for dense vertex sets.
        // For 'n' up to 100, an array/vector might be slightly simpler, 
        // but the map is flexible.
        unordered_map<int, int> distance_map;
        
        // Initialize all vertices to have 'infinity' distance.
        // A simple large number like 1e9 or INT_MAX can represent infinity for this problem
        // since max path weight is 10 * (100-1) < 1000.
        const int INF = 1e9; 
        
        // Initialize distances for all vertices from 0 to n-1
        for (int i = 0; i < n; ++i) {
            distance_map[i] = INF;
        }

        // The distance from the source to itself is 0.
        distance_map[src] = 0;

        // Min-Priority Queue (Min-Heap): Stores {distance, vertex}.
        // The default C++ priority_queue is a max-heap, so we use 'greater' 
        // for a min-heap structure.
        // Pair structure: {distance_from_src, vertex_id}
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        // Push the starting node {0, src} into the priority queue.
        pq.push({0, src});

        // --- 3. Dijkstra's Algorithm Core Loop ---
        while (!pq.empty()) {
            // Get the vertex 'u' with the smallest current distance 'd_u'
            int d_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            // Optimization: If the extracted distance is greater than the current 
            // known shortest distance, it means we found a shorter path to 'u' previously. 
            // Skip this stale entry.
            if (d_u > distance_map[u]) {
                continue;
            }

            // Iterate through all neighbors (v) of the current vertex (u)
            if (adj.count(u)) {
                for (const auto& edge : adj.at(u)) {
                    int v = edge.first;   // Destination vertex
                    int w = edge.second;  // Edge weight (u -> v)
                    
                    // Relaxation Step: Check if a shorter path to 'v' is possible 
                    // through 'u'.
                    // New path distance = (shortest distance to u) + (weight of u -> v)
                    int new_dist = d_u + w;

                    if (new_dist < distance_map[v]) {
                        // Found a shorter path! Update the distance.
                        distance_map[v] = new_dist;
                        
                        // Push the new shorter path {new_dist, v} into the priority queue.
                        pq.push({new_dist, v});
                    }
                }
            }
        }

        // --- 4. Final Result Mapping ---
        unordered_map<int, int> result;
        for (int i = 0; i < n; ++i) {
            // If the distance is still INF, the vertex is unreachable. Set distance to -1.
            if (distance_map[i] == INF) {
                result[i] = -1;
            } else {
                // Otherwise, store the calculated shortest distance.
                result[i] = distance_map[i];
            }
        }

        return result;
    }
};

