#include "_common.hpp"
#include "timer.hpp"
#include <omp.h>

inline uint8 saturate_cast(int v) {
    v = v > 0 ? v : 0;
    v = v > 255 ?  255 : v;
    return v;
}

void resize(uint8* image, shape s, uint8* data, int width, int height, Rect box, pixel mean) {

    auto get_pixel = [&image, &s, &mean](int h, int w) -> pixel* {
        if (h < 0 || h >= s.dims[0] || w < 0 || w >= s.dims[1]) {
            return &mean;
        }
        pixel* p = ((pixel*)image) + (h * s.dims[1] + w);
        return p;
    };

    // auto saturate_cast = [](int v) -> uint8 {
    //     v = v > 0 ? v : 0;
    //     v = v > 255 ? 255 : v;
    //     return v;
    // };


    float scale_x = (float)width / (box.x2 - box.x1);
    float scale_y = (float)height / (box.y2 - box.y1);

    int spatial_dim = width * height;

    uint8 *data_b, *data_g, *data_r;
    data_b = data;
    data_g = data + spatial_dim;
    data_r = data + spatial_dim * 2;
    
    int thread = omp_get_num_procs();
    if (height < thread) thread = height;
    omp_set_num_threads(thread);
#pragma omp parallel for
    for (int dst_h = 0; dst_h < height; ++ dst_h) {
        uint8 *b = data_b + dst_h * width;
        uint8 *g = data_g + dst_h * width;
        uint8 *r = data_r + dst_h * width;
        for (int dst_w = 0; dst_w < width; ++ dst_w) {
            float src_h = dst_h / scale_y + box.y1;
            float src_w = dst_w / scale_x + box.x1;

            int h1 = src_h;
            int h2 = h1 + 1;
            int w1 = src_w;
            int w2 = w1 + 1;

            pixel *p[4];

            p[0] = get_pixel(h1, w1);
            p[1] = get_pixel(h1, w2);
            p[2] = get_pixel(h2, w1);
            p[3] = get_pixel(h2, w2);

#define interp(color) \
        saturate_cast( \
            p[0]->color * (w2 - src_w) * (h2 - src_h) + \
            p[1]->color * (src_w - w1) * (h2 - src_h) + \
            p[2]->color * (w2 - src_w) * (src_h - h1) + \
            p[3]->color * (src_w - w1) * (src_h - h1)   \
            )

            *(b ++) = interp(b);
            *(g ++) = interp(g);
            *(r ++) = interp(r);
            
            // uint8 b = interp(b), g = interp(g), r = interp(r);

            // data[spatial_dim * 0 + offset] = b;
            // data[spatial_dim * 1 + offset] = g;
            // data[spatial_dim * 2 + offset] = r;
            // printf("%d %d (%hhu,%hhu,%hhu)\n", dst_h, dst_w, b, g, r);

            // offset ++;
        }
    }
}


