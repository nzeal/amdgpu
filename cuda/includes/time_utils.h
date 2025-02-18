#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

// Function to get the current time in seconds
inline double getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(now.time_since_epoch());
    return duration.count();
}

#endif // TIME_UTILS_H

