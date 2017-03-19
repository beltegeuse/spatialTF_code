#pragma once

#include <mitsuba/core/bitmap.h>

MTS_NAMESPACE_BEGIN

typedef struct {
    Float r,g,b;
} SColor;


SColor GetColour(Float v, Float vmin,Float vmax)
{
    SColor c = {1.0,1.0,1.0}; // white
    float dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + (Float)0.25 * dv)) {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    } else if (v < (vmin + (Float)0.5 * dv)) {
        c.r = 0;
        c.b = 1 + 4 * (vmin + (Float)0.25 * dv - v) / dv;
    } else if (v < (vmin + (Float)0.75 * dv)) {
        c.r = 4 * (v - vmin - (Float)0.5 * dv) / dv;
        c.b = 0;
    } else {
        c.g = 1 + 4 * (vmin + (Float)0.75 * dv - v) / dv;
        c.b = 0;
    }

    return(c);
}

void convertFalseColor(Bitmap* img, Float vMin, Float vMax) {
	Spectrum *timgPtr = (Spectrum *) img->getUInt8Data();
	for(size_t i = 0; i < img->getPixelCount(); i++) {
		Float lum = timgPtr[i].getLuminance();
		SColor c = GetColour(lum, vMin, vMax);
		timgPtr[i].fromSRGB(c.r,c.g,c.b);
	}
}

MTS_NAMESPACE_END
