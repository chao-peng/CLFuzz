
#define F_INIT   0
#define F_UPDATE 1
#define F_RESIZE 2


/*
  Parameters:
   0 pos             = the position array
   1 vel             = the velocity array
   2 iFunction       = the type of iFunction (F_INIT, F_UPDATE, F_RESIZE)
   3 uiParticleCount = the amount of particles
   4 fDeltaTime      = the delta time since last call
   5 f4BoundingBox   = the bounding box
   6 fGravity        = the fGravity
   7 iForceActive    = if the force is active (via key or button)
   8 fForce          = if the amount of force
   9 f4ForcePos      = the x,y,z position of the force point
 */
__kernel void particle_kernel(__global float4* pos,
                              __global float4* vel,
                              const int iFunction,
                              const unsigned int uiParticleCount,
                              const float fDeltaTime,
                              const float4 f3BoundingBox,
                              const float fGravity,
                              const int iForceActive,
                              const float fForce,
                              const float4 f4ForcePos
                              , __global int* ocl_kernel_branch_triggered_recorder)
{__local int my_ocl_kernel_branch_triggered_recorder[34];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 34; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // get the index in the array
  unsigned int i = get_global_id(0);

  // store the position and the velocity for this iteration
  float4 p = pos[i];
  float4 v = vel[i];

  if(iFunction == F_INIT)
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

    //figure partcles wide and tall
	  //float ny = sqrt(uiParticleCount*f3BoundingBox.y/f3BoundingBox.x);
		//float nx = uiParticleCount/ny;

    //float x = 2*(i - nx*(int)(i/nx))*f3BoundingBox.x/nx - f3BoundingBox.x;
    //float y = 2*(int)(i/nx)*f3BoundingBox.y/ny - f3BoundingBox.y;

    // calc the cubic root of particle count
    float n = rootn((float)uiParticleCount, 3);
    float dx = f3BoundingBox.x/(n+1.0f);
    float dy = f3BoundingBox.y/(n+1.0f);
    float dz = f3BoundingBox.z/(n+1.0f);


    float x =      i                % (int)n * dx+dx-(f3BoundingBox.x/2);
    float y = ((int)(i/n))          % (int)n * dy+dy-(f3BoundingBox.y/2);
    float z = ((int)(i/pown(n,2)))  % (int)n * dz+dz-(f3BoundingBox.z/2);



    //Semi-random velocities
    float vx = cos( (float)(1234*i) );
    float vy = sin( (float)(1234*i) );
    float vz = cos( (float)(1234*i) );

    //Save position and velocity
    p.x = x;
    p.y = y;
    p.z = z;

    v.x = vx;
    v.y = vy;
    v.z = vz;
  }
  else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

if(iFunction == F_UPDATE)
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);

    //Update Particle positions
    p.x += v.x*fDeltaTime;
    p.y += v.y*fDeltaTime;
    p.z += v.z*fDeltaTime;

    //Add in fGravity
    v.y -= fGravity*fDeltaTime;

    float bbx_half = f3BoundingBox.x/2;
    float bby_half = f3BoundingBox.y/2;
    float bbz_half = f3BoundingBox.z/2;

    //Edge Collision Detection
    if(p.x < -bbx_half)		//LEFT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);

      p.x = -2*bbx_half-p.x;
      v.x *= -0.9f;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);

if(p.x > bbx_half)	//RIGHT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

      p.x = 2*bbx_half-p.x;
      v.x *= -0.9f;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}
}


    if(p.y < -bby_half)		//BOTTOM
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);

      p.y  = -2*bby_half -p.y;
      v.y *= -0.45;			// if its on the bottom we extra dampen
      v.x *= 0.9;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);

if(p.y > bby_half)	//TOP
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

      p.y = 2*bby_half-p.y;
      v.y *= -0.9f;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}
}


    if(p.z < -bbz_half) // FRONT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

      p.z = -2*bbz_half-p.z;
      v.z *= -0.9f;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);

if(p.z > bbz_half) // BACK
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

      p.z = 2*bbz_half-p.z;
      v.z *= -0.9f;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}
}


    //include force
    if(iForceActive != 0)
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);

      float dx = f4ForcePos.x - p.x;
      float dy = f4ForcePos.y - p.y;
      float dz = f4ForcePos.z - p.z;

      float dist = sqrt((dx*dx+dy*dy+dz*dz));
      if(dist <1)
        {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);
dist = 1;
} else { 
atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);
}
				//This line prevents anything that is really close from getting a huge force
      v.y += dy/dist*fForce/dist;	//Use linear falloff
      v.x += dx/dist*fForce/dist;	//Exponential gives sort of a ball where the mouse is and thats it
      v.z += dz/dist*fForce/dist;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);
}

  }
  else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);

if(iFunction == F_RESIZE)
  {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[20], 1);

    float bbx_half = f3BoundingBox.x/2;
    float bby_half = f3BoundingBox.y/2;
    float bbz_half = f3BoundingBox.z/2;

    //Move everything inside box
    if(p.x < -bbx_half)		//LEFT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[22], 1);

      v.x *= -0.95f;
      p.x = -bbx_half + v.x;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[23], 1);

if(p.x > bbx_half)	//RIGHT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[24], 1);

      v.x *= -0.95f;
      p.x = bbx_half + v.x;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[25], 1);
}
}


    if(p.y < -bby_half)		//BOTTOM
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[26], 1);

      v.y *= -0.45;			// if its on the bottom we extra dampen
      v.x *= 0.95;
      p.y = -bby_half + v.y;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[27], 1);

if(p.y > bby_half)	//TOP
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[28], 1);

      v.y *=-0.95f;
      p.y = bby_half + v.y;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[29], 1);
}
}


    if(p.z < -bbz_half) // FRONT
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[30], 1);

      p.z = -2*bbz_half-p.z;
      v.z *= -0.9f;
    }
    else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[31], 1);

if(p.z > bbz_half) // BACK
    {
atomic_or(&my_ocl_kernel_branch_triggered_recorder[32], 1);

      p.z = 2*bbz_half-p.z;
      v.z *= -0.9f;
    }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[33], 1);
}
}

  }else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[21], 1);
}
}
}






  // update the arrays with our newly computed values
  pos[i] = p;
  vel[i] = v;
for (int update_recorder_i = 0; update_recorder_i < 34; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
}
