#include <vector>
#include <string>
#include <algorithm> // Not strictly necessary for Insertion Sort logic, but good practice

using namespace std;

// The Pair definition should be REMOVED here, as it's causing the redefinition error.
// The provided environment already has something like:
// struct Pair { int key; string value; };
// or
// class Pair { public: int key; string value; Pair(int key, string value); };


class Solution {
public:
    /**
     * @brief Sorts a vector of Pair objects using Insertion Sort and returns 
     * the state of the array after each insertion.
     * @param pairs The list of key-value pairs to be sorted.
     * @return vector<vector<Pair>> A list of intermediate states of the array.
     */
    vector<vector<Pair>> insertionSort(vector<Pair>& pairs) {
        vector<vector<Pair>> intermediate_states;
        int n = pairs.size();

        if (n == 0) {
            return intermediate_states;
        }

        // Loop to process each element for insertion. Starts from i=0 to record the initial state,
        // and ensures n total states are recorded (one after each major step).
        for (int i = 0; i < n; ++i) {
            
            // If i=0, we only record the state. Insertion logic starts effectively from i=1.
            if (i > 0) {
                Pair current_pair = pairs[i];
                int j = i - 1;

                // Inner loop performs comparison and shifting
                // Use strict inequality (>) to ensure stability:
                // An element equal to 'current_pair' will not be shifted, so 'current_pair' 
                // is inserted after it, preserving their relative order.
                while (j >= 0 && pairs[j].key > current_pair.key) {
                    pairs[j + 1] = pairs[j];
                    j--;
                }

                // Insert the current_pair into its correct sorted position
                pairs[j + 1] = current_pair;
            }

            // Store the state of the array after this insertion step
            intermediate_states.push_back(pairs);
        }

        return intermediate_states;
    }
};
