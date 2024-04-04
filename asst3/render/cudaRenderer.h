#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

class CudaRenderer : public CircleRenderer {

private:
  // on cpu
  Image *image;
  SceneName sceneName;
  
  int numCircles;
  float *position;
  float *velocity;
  float *color;
  float *radius;
  
  // on cuda
  float *cudaDevicePosition;
  float *cudaDeviceVelocity;
  float *cudaDeviceColor;
  float *cudaDeviceRadius;
  float *cudaDeviceImageData;
  bool *pixelMarkedArray;

public:
  CudaRenderer();
  virtual ~CudaRenderer();

  const Image *getImage();

  void setup();

  void loadScene(SceneName name);

  void allocOutputImage(int width, int height);

  void clearImage();

  void advanceAnimation();

  void render();

};

#endif
