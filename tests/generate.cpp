#include "defines.h"
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>


float rnd(float rnd_min, float rnd_max) 
{
    return rnd_min + (rnd_max - rnd_min) * rand() / (float) RAND_MAX;
}

std::vector<float> generate_image(int side) 
{
    std::cout << "Generating " << side << "...\n";

    int all = side * side;
    std::vector<float> data(all);

    float kx = side * 0.2f;
    float ky = side * 0.3f;
    float kxy = side * 0.12f;
    float kyx = side * 0.33f;

    srand((int)time(NULL));

    struct Center {
        int x, y; // position
        float a;  // amplitude
        float p;  // phase multiplier
    };

    Center centers[RND_CENTERS_C];

    // generate some ripple centers
    for( int i = 0; i < RND_CENTERS_C; i++) {
        centers[i].x = rnd(0.0f, side);
        centers[i].y = rnd(0.0f, side);
        centers[i].a = rnd(0.0f, RND_AMP);
        centers[i].p = rnd(0.0f, RND_PHASE);
    }

    float offset_x = rnd(0.0f, side);
    float offset_y = rnd(0.0f, side);

    for ( int y = 0; y < side; ++y)
    {
        for (int x = 0; x < side; ++x)
        {
            int ox = x - offset_y;
            int oy = y - offset_y;

            float val = 0.0f;
            for( int i = 0; i < RND_CENTERS_C; i++) {
                Center&  c = centers[i];
                val += sinf((x - c.x)*c.p) * cosf((y - c.y)*c.p) * c.a;
            }
            data[y*side + x] = val;
        }
    }

    return data;
}


// normalize it [0..1]
void normalize(int side, std::vector<float>& data) 
{
    std::cout << "Normalizing " << side << "...\n";
    int all = side * side;
    float max_val = -MAX_FLT;
    float min_val =  MAX_FLT;
    for( int i = 0; i < all; i ++) {
        if( min_val > data[i] ) {
            min_val = data[i];
        }

        if( max_val < data[i] ) {
            max_val = data[i];
        }
    }

    float diff = max_val - min_val;
    float scale_k = 1.0f / diff;

    for( int i = 0; i < all; i ++) {
        data[i] = (data[i] - min_val) * scale_k;
    }
}

// To previe generated file
bool write_ppm(const char* path, int side, std::vector<float>& data) 
{
    std::cout << "Saving " << path << "...\n";
    int all = side * side;

    FILE *fp = fopen(path, "wb");

    if( !fp) {
        return false;
    }

    (void) fprintf(fp, "P6\n%d %d\n255\n", side, side);
    for ( int y = 0; y < side; ++y)
    {
        for (int x = 0; x < side; ++x)
        {
            static unsigned char color[3];
            int index = y*side + x; // no need to optimize it
            float val = data[index];

            int intensity = 255 * val;
            color[0] = val;
            color[1] = val;
            color[2] = val;
            if( !fwrite(color, 1, 3, fp) ) {
                return false;
            }
        }
    }
    (void) fclose(fp);
    
    return true;
}