#include "tests.h"
#include <filesystem>
#include <iostream>

int main() {
    std::cout << "==================================="  << std::endl;
    std::cout << "Starting SoftServe task in"  << std::filesystem::current_path() << std::endl;
    std::cout << "==================================="  << std::endl;

    // all_correctness();
    all_benchmarks();

    std::cout << "Softserve task finished!" << std::endl;
    std::cout << "==================================="  << std::endl;
    return 0;
}

