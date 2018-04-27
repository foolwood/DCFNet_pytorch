#ifndef _TIMER_HPP_
#define _TIMER_HPP_
#include "time.h"
#include <sys/time.h>
inline static double getMSeconds(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000.+tv.tv_usec/1000.;
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1): \
        strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1):__FILE__)

// #define SHOW_TIME

#ifdef SHOW_TIME

#define TS(mark) \
    double st##mark = getMSeconds()

#define TE(mark) \
    do { \
        double ed = getMSeconds(); \
        printf("[%s,%d] %s:%f ms\n", __FILENAME__, __LINE__, #mark, ed - st##mark); \
    } while(0)

#else

#define TS(...)

#define TE(...)

#endif

#endif
