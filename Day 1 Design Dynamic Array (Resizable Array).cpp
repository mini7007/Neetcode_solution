#include <iostream>
#include <stdexcept>

class DynamicArray {
private:
    // Raw pointer to the dynamically allocated array
    int* data;
    // Current number of elements stored
    int size;
    // Total space allocated
    int capacity;

public:
    /**
     * @brief Constructor: Initializes the dynamic array.
     */
    DynamicArray(int initial_capacity) {
        if (initial_capacity <= 0) {
            // Ensure valid capacity
            throw std::invalid_argument("Capacity must be greater than 0.");
        }
        this->capacity = initial_capacity;
        this->size = 0;
        // Allocate the initial memory on the heap
        this->data = new int[this->capacity];
    }

    /**
     * @brief Destructor: Releases the allocated memory to prevent leaks.
     */
    ~DynamicArray() {
        delete[] this->data;
    }

    /**
     * @brief Returns the element at the specified index.
     * @note Index is guaranteed to be valid.
     */
    int get(int i) {
        return this->data[i];
    }

    /**
     * @brief Sets the element at the specified index.
     * @note Index is guaranteed to be valid.
     */
    void set(int i, int n) {
        this->data[i] = n;
    }

    /**
     * @brief Adds an element to the end of the array, resizing if necessary.
     */
    void pushback(int n) {
        if (this->size == this->capacity) {
            resize();
        }
        this->data[this->size] = n;
        this->size++;
    }

    /**
     * @brief Removes and returns the element at the end of the array.
     * @note Array is guaranteed to be non-empty.
     */
    int popback() {
        // Decrease size first, then return the element at the old 'size' index
        this->size--;
        return this->data[this->size];
    }

    /**
     * @brief Doubles the capacity of the array and copies all elements.
     */
    void resize() {
        int newCapacity = this->capacity * 2;
        
        // 1. Allocate a new, larger array
        int* newArray = new int[newCapacity];

        // 2. Copy the elements
        for (int i = 0; i < this->size; ++i) {
            newArray[i] = this->data[i];
        }

        // 3. Delete the old array memory
        delete[] this->data;

        // 4. Update the pointer and capacity
        this->data = newArray;
        this->capacity = newCapacity;
    }

    /**
     * @brief Returns the current number of elements.
     */
    int getSize() {
        return this->size;
    }

    /**
     * @brief Returns the total allocated capacity.
     */
    int getCapacity() {
        return this->capacity;
    }
};
