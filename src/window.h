/**
 * CPU queue objects.
 * Based on Scott Beamer's gapbs/src/sliding_queue.h
 */

#ifndef SRC__WINDOW_H
#define SRC__WINDOW_H

#include <algorithm>
#include <cstddef>

template <typename T>
class LocalWindow;

/**
 * Unsafe sliding queue.
 * NOT THREAD SAFE. DOESN"T CHECK BUFFER OVERFLOW.
 */
template <typename T>
class SlidingWindow {
public:
    SlidingWindow(size_t buffer_size);
    ~SlidingWindow();

    void push_back(T val);
    void slide_window();
    void reset();

    using iterator = T*;
    bool     empty() const { return current_start == current_end; }
    iterator begin() const { return buffer + current_start; }
    iterator end()   const { return buffer + current_end; }
    size_t   size()  const { return end() - begin(); }

    friend LocalWindow<T>;

private:
    T           *buffer;       // Buffer.
    std::size_t current_start; // Current window start.
    std::size_t current_end;   // Current window end (exclusive).
    std::size_t next_end;      // Next window end (exclusive).
};

/**
 * Local CPU next window queues.
 * NOT THREAD SAFE.
 */
template <typename T>
class LocalWindow {
public:
    LocalWindow(SlidingWindow<T> &parent_, size_t local_buffer_size = 16384);
    ~LocalWindow();

    void push_back(T val);
    void flush();

private:
    SlidingWindow<T> &parent;
    T               *buffer;
    size_t          cur_size;
    const size_t    buffer_size;
};

/*****************************************************************************
 ***** Implementation ********************************************************
 *****************************************************************************/

template <typename T>
SlidingWindow<T>::SlidingWindow(size_t buffer_size) {
    buffer = new T[buffer_size];
    reset();
}

template <typename T>
SlidingWindow<T>::~SlidingWindow() {
    delete[] buffer;
}

/**
 * Push back value into next window. 
 * Parameters:
 *   val <- next window's value.
 */
template <typename T>
void SlidingWindow<T>::push_back(T val) {
    buffer[next_end++] = val;
}

/**
 * Slides window from current to next.
 */
template <typename T>
void SlidingWindow<T>::slide_window() {
    current_start = current_end;
    current_end   = next_end;
}

/**
 * Resets sliding window to initial position.
 */
template <typename T>
void SlidingWindow<T>::reset() {
    current_start = current_end = next_end = 0;
}

template <typename T>
LocalWindow<T>::LocalWindow(SlidingWindow<T> &parent_, size_t local_buffer_size)
    : parent(parent_)
    , buffer(new T[local_buffer_size])
    , cur_size(0)
    , buffer_size(local_buffer_size)
{}

template <typename T>
LocalWindow<T>::~LocalWindow() {
    delete[] buffer;
}

/**
 * Push back value into local window.
 * Updates shared window if necessary.
 * 
 * Parameters:
 *   val <- value for window.
 */
template <typename T>
void LocalWindow<T>::push_back(T val) {
    //  Flush if local buffer is full.
    if (cur_size == buffer_size)
        flush();
    buffer[cur_size++] = val;
}

/**
 * Pushes local value to parent's next window.
 */
template <typename T>
void LocalWindow<T>::flush() {
    T *parent_buffer = parent.buffer;
    size_t copy_start = __sync_fetch_and_add(&parent.next_end, cur_size);
    std::copy(buffer, buffer + cur_size, parent_buffer + copy_start);
    cur_size = 0;
}

#endif // SRC__WINDOW_H
