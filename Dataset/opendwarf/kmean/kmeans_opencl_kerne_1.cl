#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
//texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
//texture<float, 1, cudaReadModeElementType> t_features_flipped;
//texture<float, 1, cudaReadModeElementType> t_clusters;


//__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ...
   [p1,dim0][p1,dim1][p1,dim2] ...
   [p2,dim0][p2,dim1][p2,dim2] ...
   to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
 */
__kernel void invert_mapping(__global float *input,
		__global float *output,
		int npoints,
		int nfeatures)
{
	int point_id = get_local_id(0) + get_local_size(0)*get_group_id(0);
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
	}
	return;
}


#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
