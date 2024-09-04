#ifndef COMMON_H
#define COMMON_H

#include <sys/time.h>

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) tp.tv_sec + (double) tp.tv_usec * 1e-6;
}

#endif