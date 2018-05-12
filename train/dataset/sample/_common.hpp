#ifndef __COMMON_HPP_
#define __COMMON_HPP_

typedef struct shape {
    int ndim;
    long int*dims;
} shape;

typedef struct Rect {
    float x1, y1, x2, y2;
} Rect;


typedef unsigned char uint8;

typedef struct pixel {
    uint8 b, g, r;
} pixel;

typedef struct pixelf {
    float b, g, r;
} pixelf;


#endif
