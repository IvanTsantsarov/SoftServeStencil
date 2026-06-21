#include "tests.h"
#include <filesystem>
#include <iostream>

int main() {
    std::cout << "==================================="  << '\n';
    std::cout << "Starting SoftServe task in"  << std::filesystem::current_path() << '\n';
    std::cout << "==================================="  << '\n';

    all_correctness();
    all_benchmarks();

    std::cout << "Softserve task finished!" << '\n';
    std::cout << "==================================="  << '\n';
    return 0;
}

