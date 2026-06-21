#pragma once

#include <vector>

std::vector<float> generate_image(int side);
void normalize(int side, std::vector<float>& data);
bool write_ppm(const char* path, int side, std::vector<float>& data);

bool all_correctness();
bool all_benchmarks();
