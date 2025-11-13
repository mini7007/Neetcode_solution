#include <vector>
#include <iostream>

using namespace std;

class LinkedList {
private:
    // Define the Node structure inside the class for proper scoping
    struct Node {
        int val;
        Node* next;
        // Constructor
        Node(int v) : val(v), next(nullptr) {}
    };

    Node* head; // Pointer to the first node
    int size;   // Number of nodes in the list

public:
    // Constructor: Initializes an empty linked list
    LinkedList() : head(nullptr), size(0) {}

    // Destructor: Frees all allocated memory to prevent memory leaks
    ~LinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

    // Returns the value of the ith node (0-indexed).
    // If the index is out of bounds, returns -1.
    int get(int index) {
        if (index < 0 || index >= size) {
            return -1;
        }

        Node* current = head;
        // Traverse to the specified index
        for (int i = 0; i < index; ++i) {
            current = current->next;
        }
        return current->val;
    }

    // Inserts a node with val at the head of the list.
    void insertHead(int val) {
        Node* newNode = new Node(val);
        newNode->next = head; // New node points to the old head
        head = newNode;        // Update head to the new node
        size++;
    }

    // Inserts a node with val at the tail of the list.
    void insertTail(int val) {
        Node* newNode = new Node(val);
        
        if (head == nullptr) {
            head = newNode;
        } else {
            Node* current = head;
            // Traverse to the last node
            while (current->next != nullptr) {
                current = current->next;
            }
            current->next = newNode; // Last node points to the new tail
        }
        size++;
    }

    // Removes the ith node (0-indexed).
    // If the index is out of bounds, returns false, otherwise returns true.
    bool remove(int index) {
        if (index < 0 || index >= size) {
            return false;
        }

        Node* nodeToDelete = nullptr;

        if (index == 0) {
            // Case 1: Removing the head
            nodeToDelete = head;
            head = head->next;
        } else {
            // Case 2: Removing a node that is not the head
            Node* prev = head;
            // Traverse to the node *before* the one to be removed
            for (int i = 0; i < index - 1; ++i) {
                prev = prev->next;
            }
            
            nodeToDelete = prev->next;
            prev->next = nodeToDelete->next; // Bypass the node to be removed
        }
        
        // Safely delete the node
        if (nodeToDelete != nullptr) {
            delete nodeToDelete;
        }
        size--;
        return true;
    }

    // Returns an array (vector) of all the values in the linked list,
    // ordered from head to tail.
    vector<int> getValues() {
        vector<int> values;
        Node* current = head;
        
        while (current != nullptr) {
            values.push_back(current->val);
            current = current->next;
        }
        return values;
    }
};
