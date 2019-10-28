
// define types based on compiler "command line"
#if defined(SINGLE_PRECISION)
#define VALTYPE float
#elif defined(K_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VALTYPE double
#elif defined(AMD_DOUBLE_PRECISION)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VALTYPE double

#endif
#define VALTYPE float

inline
int
ToGlobalRow( int gidRow, int lszRow, int lidRow )
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global row (without halo)
    return gidRow*lszRow + lidRow;
}

inline
int
ToGlobalCol( int gidCol, int lszCol, int lidCol )
{
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global column (without halo)
    return gidCol*lszCol + lidCol;
}


inline
int
ToFlatHaloedIdx( int row, int col, int rowPitch )
{
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo of width 1
    return (row + 1)*(rowPitch + 2) + (col + 1);
}


inline
int
ToFlatIdx( int row, int col, int pitch )
{
    return row * pitch + col;
}


__kernel
void
CopyRect( __global VALTYPE* dest,
            int doffset,
            int dpitch,
            __global VALTYPE* src,
            int soffset,
            int spitch,
            int width,
            int height )
{
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int gsz = get_global_size(0);
    int lsz = get_local_size(0);
    int grow = gid * lsz + lid;

    if( grow < height )
    {
        for( int c = 0; c < width; c++ )
        {
            (dest + doffset)[ToFlatIdx(grow,c,dpitch)] = (src + soffset)[ToFlatIdx(grow,c,spitch)];
        }
    }
}
