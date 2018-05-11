#ifndef SDF_H
#define SDF_H

#include <vector>

struct sdf_t {
    int * ds;
    int w;
    int h;

    sdf_t(int * ds, int w, int h){
	this->ds = ds;
	this->w = w;
	this->h = h;
    }

    ~sdf_t(){ delete[] ds; }
}

#endif
