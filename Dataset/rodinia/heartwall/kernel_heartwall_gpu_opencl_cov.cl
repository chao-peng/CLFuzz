#define OCL_NEW_BARRIER(barrierid,arg)\
{\
  atom_inc(&ocl_kernel_barrier_count[barrierid]);\
  barrier(arg);\
  if (ocl_kernel_barrier_count[barrierid]!=ocl_get_general_size()) {\
    ocl_barrier_divergence_recorder[barrierid]=1;\
  }\
  barrier(arg);\
  ocl_kernel_barrier_count[barrierid]=0;\
  barrier(arg);\
}
int ocl_get_general_size(){
  int result = 1;\
  for (int i=0; i<get_work_dim(); i++){
    result*=get_local_size(i);
  }
  return result;
}

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL
//========================================================================================================================================================================================================200

__kernel void 
kernel_gpu_opencl(	// structures
					params_common d_common,					// 0

					// common_change
					__global fp* d_frame,					// 1	INPUT
					int d_frame_no,							// 2	INPUT

					// common
					__global int* d_endoRow,				// 3	INPUT
					__global int* d_endoCol,				// 4	INPUT
					__global int* d_tEndoRowLoc,			// 5	OUTPUT	common.endoPoints * common.no_frames
					__global int* d_tEndoColLoc,			// 6	OUTPUT	common.endoPoints * common.no_frames
					__global int* d_epiRow,					// 7	INPUT
					__global int* d_epiCol,					// 8	INPUT
					__global int* d_tEpiRowLoc,				// 9	OUTPUT	common.epiPoints * common.no_frames
					__global int* d_tEpiColLoc,				// 10	OUTPUT	common.epiPoints * common.no_frames

					// common_unique
					__global fp* d_endoT,					// 11	OUTPUT	common.in_elem * common.endoPoints
					__global fp* d_epiT,					// 12	OUTPUT	common.in_elem * common.epiPoints
					__global fp* d_in2_all,					// 13	OUTPUT	common.in2_elem * common.allPoints
					__global fp* d_conv_all,				// 14	OUTPUT	common.conv_elem * common.allPoints
					__global fp* d_in2_pad_cumv_all,		// 15	OUTPUT	common.in2_pad_cumv_elem * common.allPoints
					__global fp* d_in2_pad_cumv_sel_all,	// 16	OUTPUT	common.in2_pad_cumv_sel_elem * common.allPoints
					__global fp* d_in2_sub_cumh_all,		// 17	OUTPUT	common.in2_sub_cumh_elem * common.allPoints
					__global fp* d_in2_sub_cumh_sel_all,	// 18	OUTPUT	common.in2_sub_cumh_sel_elem * common.allPoints
					__global fp* d_in2_sub2_all,			// 19	OUTPUT	common.in2_sub2_elem * common.allPoints
					__global fp* d_in2_sqr_all,				// 20	OUTPUT	common.in2_elem * common.allPoints
					__global fp* d_in2_sqr_sub2_all,		// 21	OUTPUT	common.in2_sub2_elem * common.allPoints
					__global fp* d_in_sqr_all,				// 22	OUTPUT	common.in_elem * common.allPoints
					__global fp* d_tMask_all,				// 23	OUTPUT	common.tMask_elem * common.allPoints
					__global fp* d_mask_conv_all,			// 24	OUTPUT	common.mask_conv_elem * common.allPoints

					// // local
					// __local fp* d_in_mod_temp,			// 25	OUTPUT	common.in_elem
					// __local fp* in_partial_sum,			// 26	OUTPUT	common.in_cols
					// __local fp* in_sqr_partial_sum,		// 27	OUTPUT	common.in_sqr_rows
					// __local fp* par_max_val,				// 28	OUTPUT	common.mask_conv_rows
					// __local int* par_max_coo)			// 29	OUTPUT	common.mask_conv_rows

					// local
					__global fp* d_in_mod_temp_all,			// 25	OUTPUT	common.in_elem * common.allPoints
					__global fp* in_partial_sum_all,		// 26	OUTPUT	common.in_cols * common.allPoints
					__global fp* in_sqr_partial_sum_all,	// 27	OUTPUT	common.in_sqr_rows * common.allPoints
					__global fp* par_max_val_all,			// 28	OUTPUT	common.mask_conv_rows * common.allPoints
					__global int* par_max_coo_all,			// 29	OUTPUT	common.mask_conv_rows * common.allPoints

					__global fp* in_final_sum_all,			// 30	OUTPUT	common.allPoints
					__global fp* in_sqr_final_sum_all,		// 31	OUTPUT	common.allPoints
					__global fp* denomT_all,				// 32	OUTPUT	common.allPoints

					__global fp* checksum, __global int* ocl_kernel_branch_triggered_recorder, __global int* ocl_barrier_divergence_recorder, __global int* ocl_kernel_loop_recorder)					// 33	OUTPUT	100

{__local int my_ocl_kernel_branch_triggered_recorder[82];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 82; ++ocl_kernel_init_i) {
    my_ocl_kernel_branch_triggered_recorder[ocl_kernel_init_i] = 0;
}
__local int ocl_kernel_barrier_count[41];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 41; ++ocl_kernel_init_i) {
    ocl_kernel_barrier_count[ocl_kernel_init_i] = 0;
}
__local int my_ocl_kernel_loop_recorder[48];
for (int ocl_kernel_init_i = 0; ocl_kernel_init_i < 48; ++ocl_kernel_init_i) {
    my_ocl_kernel_loop_recorder[ocl_kernel_init_i] = 0;
}
int private_ocl_kernel_loop_iter_counter[48];
bool private_ocl_kernel_loop_boundary_not_reached[48];
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	//======================================================================================================================================================150
	//	COMMON VARIABLES
	//======================================================================================================================================================150

	// __global fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	fp s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	fp sum;
	int pos_ori;
	fp temp;
	fp temp2;
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp mean;
	fp mean_sqr;
	fp variance;
	fp deviation;
	int pointer;
	int ori_pointer;
	int loc_pointer;

	// __local fp in_final_sum;
	// __local fp in_sqr_final_sum;
	// __local fp denomT;

	//======================================================================================================================================================150
	//	BLOCK/THREAD IDs
	//======================================================================================================================================================150

	int bx = get_group_id(0);															// get current horizontal block index (0-n)
	int tx = get_local_id(0);															// get current horizontal thread index (0-n)
	int ei_new;

	//======================================================================================================================================================150
	//	UNIQUE STRUCTURE RECONSTRUCTED HERE
	//======================================================================================================================================================150

	// common
	__global fp* d_common_change_d_frame = &d_frame[0];

	// offsets for either endo or epi points (separate arrays for endo and epi points)
	int d_unique_point_no;
	__global int* d_unique_d_Row;
	__global int* d_unique_d_Col;
	__global int* d_unique_d_tRowLoc;
	__global int* d_unique_d_tColLoc;
	__global fp* d_in;
	if(bx < d_common.endoPoints){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[0], 1);

		d_unique_point_no = bx;													// endo point number 0-???
		d_unique_d_Row = d_endoRow;												// initial endo row coordinates
		d_unique_d_Col = d_endoCol;												// initial endo col coordinates
		d_unique_d_tRowLoc = d_tEndoRowLoc;										// all endo row coordinates
		d_unique_d_tColLoc = d_tEndoColLoc;										// all endo col coordinates
		d_in = &d_endoT[d_unique_point_no * d_common.in_elem];					// endo templates
	}
	else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[1], 1);

		d_unique_point_no = bx-d_common.endoPoints;								// epi point number 0-???
		d_unique_d_Row = d_epiRow;												// initial epi row coordinates
		d_unique_d_Col = d_epiCol;												// initial epi col coordinates
		d_unique_d_tRowLoc = d_tEpiRowLoc;										// all epi row coordinates
		d_unique_d_tColLoc = d_tEpiColLoc;										// all epi col coordinates
		d_in = &d_epiT[d_unique_point_no * d_common.in_elem];					// epi templates
	}

	// offsets for all points (one array for all points)
	__global fp* d_unique_d_in2 = &d_in2_all[bx*d_common.in2_elem];
	__global fp* d_unique_d_conv = &d_conv_all[bx*d_common.conv_elem];
	__global fp* d_unique_d_in2_pad_cumv = &d_in2_pad_cumv_all[bx*d_common.in2_pad_cumv_elem];
	__global fp* d_unique_d_in2_pad_cumv_sel = &d_in2_pad_cumv_sel_all[bx*d_common.in2_pad_cumv_sel_elem];
	__global fp* d_unique_d_in2_sub_cumh = &d_in2_sub_cumh_all[bx*d_common.in2_sub_cumh_elem];
	__global fp* d_unique_d_in2_sub_cumh_sel = &d_in2_sub_cumh_sel_all[bx*d_common.in2_sub_cumh_sel_elem];
	__global fp* d_unique_d_in2_sub2 = &d_in2_sub2_all[bx*d_common.in2_sub2_elem];
	__global fp* d_unique_d_in2_sqr = &d_in2_sqr_all[bx*d_common.in2_sqr_elem];
	__global fp* d_unique_d_in2_sqr_sub2 = &d_in2_sqr_sub2_all[bx*d_common.in2_sqr_sub2_elem];
	__global fp* d_unique_d_in_sqr = &d_in_sqr_all[bx*d_common.in_sqr_elem];
	__global fp* d_unique_d_tMask = &d_tMask_all[bx*d_common.tMask_elem];
	__global fp* d_unique_d_mask_conv = &d_mask_conv_all[bx*d_common.mask_conv_elem];

	// used to be local
	__global fp* d_in_mod_temp = &d_in_mod_temp_all[bx*d_common.in_elem];
	__global fp* in_partial_sum = &in_partial_sum_all[bx*d_common.in_cols];
	__global fp* in_sqr_partial_sum = &in_sqr_partial_sum_all[bx*d_common.in_sqr_rows];
	__global fp* par_max_val = &par_max_val_all[bx*d_common.mask_conv_rows];
	__global int* par_max_coo = &par_max_coo_all[bx*d_common.mask_conv_rows];

	__global fp* in_final_sum = &in_final_sum_all[bx];
	__global fp* in_sqr_final_sum = &in_sqr_final_sum_all[bx];
	__global fp* denomT = &denomT_all[bx];

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	Initialize checksum
	//======================================================================================================================================================150
#ifdef TEST_CHECKSUM
	if(bx==0 && tx==0){

		for(i=0; i<CHECK; i++){
			checksum[i] = 0;
		}

	}
#endif
	//======================================================================================================================================================150
	//	INITIAL COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	// generate templates based on the first frame only
	if(d_frame_no == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[2], 1);


		//====================================================================================================100
		//	Initialize cross-frame variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block and for all frames
			for(i=0; i<d_common.no_frames; i++){
				d_unique_d_tRowLoc[d_unique_point_no*d_common.no_frames+i] = 0;
				d_unique_d_tColLoc[d_unique_point_no*d_common.no_frames+i] = 0;
			}

			// this block
			for(i=0; i<d_common.in_elem; i++){
				d_in[i] = 0;
			}

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		OCL_NEW_BARRIER(0,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	UPDATE ROW LOC AND COL LOC
		//====================================================================================================100

		// uptade temporary endo/epi row/col coordinates (in each block corresponding to point, narrow work to one thread)
		ei_new = tx;
		if(ei_new == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[4], 1);


			// update temporary row/col coordinates
			pointer = d_unique_point_no*d_common.no_frames+d_frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no];
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no];

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[5], 1);
}

		//====================================================================================================100
		//	CREATE TEMPLATES
		//====================================================================================================100

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[0] = 0;
private_ocl_kernel_loop_boundary_not_reached[0] = true;
while(ei_new < d_common.in_elem || (private_ocl_kernel_loop_boundary_not_reached[0] = false)){
private_ocl_kernel_loop_iter_counter[0]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[6], 1);

				row = d_common.in_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[7], 1);
}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[col*d_common.in_rows+row] = d_common_change_d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[0] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 1);
}if (private_ocl_kernel_loop_iter_counter[0] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 2);
}if (private_ocl_kernel_loop_iter_counter[0] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[0]) {
    atomic_or(&my_ocl_kernel_loop_recorder[0], 8);
}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		OCL_NEW_BARRIER(1,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	CHECKSUM
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[0] = checksum[0]+d_in[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//====================================================================================================100
		//	End
		//====================================================================================================100

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[3], 1);
}

	//======================================================================================================================================================150
	//	PROCESS POINTS
	//======================================================================================================================================================150

	// process points in all frames except for the first one
	if(d_frame_no != 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[8], 1);


		//====================================================================================================100
		//	Initialize frame-specific variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block
			for(i=0; i<d_common.in2_elem; i++){
				d_unique_d_in2[i] = 0;
			}
			for(i=0; i<d_common.conv_elem; i++){
				d_unique_d_conv[i] = 0;
			}
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				d_unique_d_in2_pad_cumv[i] = 0;
			}
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				d_unique_d_in2_pad_cumv_sel[i] = 0;
			}
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				d_unique_d_in2_sub_cumh[i] = 0;
			}
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				d_unique_d_in2_sub_cumh_sel[i] = 0;
			}
			for(i=0; i<d_common.in2_sub2_elem; i++){
				d_unique_d_in2_sub2[i] = 0;
			}
			for(i=0; i<d_common.in2_sqr_elem; i++){
				d_unique_d_in2_sqr[i] = 0;
			}
			for(i=0; i<d_common.in2_sqr_sub2_elem; i++){
				d_unique_d_in2_sqr_sub2[i] = 0;
			}
			for(i=0; i<d_common.in_sqr_elem; i++){
				d_unique_d_in_sqr[i] = 0;
			}
			for(i=0; i<d_common.tMask_elem; i++){
				d_unique_d_tMask[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_elem; i++){
				d_unique_d_mask_conv[i] = 0;
			}

			for(i=0; i<d_common.in_elem; i++){
				d_in_mod_temp[i] = 0;
			}
			for(i=0; i<d_common.in_cols; i++){
				in_partial_sum[i] = 0;
			}
			for(i=0; i<d_common.in_sqr_rows; i++){
				in_sqr_partial_sum[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_rows; i++){
				par_max_val[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_rows; i++){
				par_max_coo[i] = 0;
			}

			in_final_sum[0] = 0;
			in_sqr_final_sum[0] = 0;
			denomT[0] = 0;

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		OCL_NEW_BARRIER(2,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	SELECTION
		//====================================================================================================100

		in2_rowlow = d_unique_d_Row[d_unique_point_no] - d_common.sSize;													// (1 to n+1)
		in2_collow = d_unique_d_Col[d_unique_point_no] - d_common.sSize;

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[1] = 0;
private_ocl_kernel_loop_boundary_not_reached[1] = true;
while(ei_new < d_common.in2_elem || (private_ocl_kernel_loop_boundary_not_reached[1] = false)){
private_ocl_kernel_loop_iter_counter[1]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[10], 1);

				row = d_common.in2_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[11], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_unique_d_in2[ei_new] = d_common_change_d_frame[ori_col*d_common.frame_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[1] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 1);
}if (private_ocl_kernel_loop_iter_counter[1] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 2);
}if (private_ocl_kernel_loop_iter_counter[1] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[1]) {
    atomic_or(&my_ocl_kernel_loop_recorder[1], 8);
}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		OCL_NEW_BARRIER(3,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	CHECKSUM
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_elem; i++){
				checksum[1] = checksum[1]+d_unique_d_in2[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//====================================================================================================100
		//	CONVOLUTION
		//====================================================================================================100

		//==================================================50
		//	ROTATION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[2] = 0;
private_ocl_kernel_loop_boundary_not_reached[2] = true;
while(ei_new < d_common.in_elem || (private_ocl_kernel_loop_boundary_not_reached[2] = false)){
private_ocl_kernel_loop_iter_counter[2]++;

		// while(ei_new < 1){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[12], 1);

				row = d_common.in_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[13], 1);
}

			// execution
			rot_row = (d_common.in_rows-1) - row;
			rot_col = (d_common.in_rows-1) - col;
			d_in_mod_temp[ei_new] = d_in[rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = d_in[0];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_in_pointer];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * d_common.in_elem];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * 2601];
			// if((d_unique_point_no * d_common.in_elem) > (2601*51
				// printf("frame_no IS %d\n", d_common_change[0].frame_no);
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no];

			// d_in_mod_temp[ei_new] = 1;
			// kot = d_in[rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = kot;
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_in_pointer+rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * d_common.in_elem+rot_col*d_common.in_rows+rot_row];
			//d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no];
			// d_unique_d_T[d_unique_in_pointer+rot_col*d_common.in_rows+rot_row] = 1;
			// d_unique_d_T[d_unique_in_pointer] = 1;
			// d_endoT[d_unique_in_pointer] = 1;
			// d_in_mod_temp[ei_new] = 1;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[2] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 1);
}if (private_ocl_kernel_loop_iter_counter[2] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 2);
}if (private_ocl_kernel_loop_iter_counter[2] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[2]) {
    atomic_or(&my_ocl_kernel_loop_recorder[2], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(4,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[2] = checksum[2]+d_in_mod_temp[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	ACTUAL CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[3] = 0;
private_ocl_kernel_loop_boundary_not_reached[3] = true;
while(ei_new < d_common.conv_elem || (private_ocl_kernel_loop_boundary_not_reached[3] = false)){
private_ocl_kernel_loop_iter_counter[3]++;


			// figure out row/col location in array
			ic = (ei_new+1) % d_common.conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.conv_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[14], 1);

				ic = d_common.conv_rows;
				jc = jc-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[15], 1);
}

			//
			j = jc + d_common.joffset;
			jp1 = j + 1;
			if(d_common.in2_cols < jp1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[16], 1);

				ja1 = jp1 - d_common.in2_cols;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[17], 1);

				ja1 = 1;
			}
			if(d_common.in_cols < j){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[18], 1);

				ja2 = d_common.in_cols;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[19], 1);

				ja2 = j;
			}

			i = ic + d_common.ioffset;
			ip1 = i + 1;
			
			if(d_common.in2_rows < ip1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[20], 1);

				ia1 = ip1 - d_common.in2_rows;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[21], 1);

				ia1 = 1;
			}
			if(d_common.in_rows < i){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[22], 1);

				ia2 = d_common.in_rows;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[23], 1);

				ia2 = i;
			}

			s = 0;

			private_ocl_kernel_loop_iter_counter[4] = 0;
private_ocl_kernel_loop_boundary_not_reached[4] = true;
for(ja=ja1; ja<=ja2 || (private_ocl_kernel_loop_boundary_not_reached[4] = false); ja++){
private_ocl_kernel_loop_iter_counter[4]++;

				jb = jp1 - ja;
				private_ocl_kernel_loop_iter_counter[5] = 0;
private_ocl_kernel_loop_boundary_not_reached[5] = true;
for(ia=ia1; ia<=ia2 || (private_ocl_kernel_loop_boundary_not_reached[5] = false); ia++){
private_ocl_kernel_loop_iter_counter[5]++;

					ib = ip1 - ia;
					s = s + d_in_mod_temp[d_common.in_rows*(ja-1)+ia-1] * d_unique_d_in2[d_common.in2_rows*(jb-1)+ib-1];
				}
if (private_ocl_kernel_loop_iter_counter[5] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 1);
}if (private_ocl_kernel_loop_iter_counter[5] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 2);
}if (private_ocl_kernel_loop_iter_counter[5] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[5]) {
    atomic_or(&my_ocl_kernel_loop_recorder[5], 8);
}
			}
if (private_ocl_kernel_loop_iter_counter[4] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 1);
}if (private_ocl_kernel_loop_iter_counter[4] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 2);
}if (private_ocl_kernel_loop_iter_counter[4] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[4]) {
    atomic_or(&my_ocl_kernel_loop_recorder[4], 8);
}

			//d_unique_d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_conv[ei_new] = s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[3] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 1);
}if (private_ocl_kernel_loop_iter_counter[3] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 2);
}if (private_ocl_kernel_loop_iter_counter[3] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[3]) {
    atomic_or(&my_ocl_kernel_loop_recorder[3], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(5,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[3] = checksum[3]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		// 	CUMULATIVE SUM	(LOCAL)
		//====================================================================================================100

		//==================================================50
		//	PADD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[6] = 0;
private_ocl_kernel_loop_boundary_not_reached[6] = true;
while(ei_new < d_common.in2_pad_cumv_elem || (private_ocl_kernel_loop_boundary_not_reached[6] = false)){
private_ocl_kernel_loop_iter_counter[6]++;


			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[24], 1);

				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[25], 1);
}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_cols)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[26], 1);

				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2[ori_col*d_common.in2_rows+ori_row];
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[27], 1);
																			// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[6] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 1);
}if (private_ocl_kernel_loop_iter_counter[6] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 2);
}if (private_ocl_kernel_loop_iter_counter[6] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[6]) {
    atomic_or(&my_ocl_kernel_loop_recorder[6], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(6,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[4] = checksum[4]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[7] = 0;
private_ocl_kernel_loop_boundary_not_reached[7] = true;
while(ei_new < d_common.in2_pad_cumv_cols || (private_ocl_kernel_loop_boundary_not_reached[7] = false)){
private_ocl_kernel_loop_iter_counter[7]++;


			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			private_ocl_kernel_loop_iter_counter[8] = 0;
private_ocl_kernel_loop_boundary_not_reached[8] = true;
for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows || (private_ocl_kernel_loop_boundary_not_reached[8] = false); position = position + 1){
private_ocl_kernel_loop_iter_counter[8]++;

				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}
if (private_ocl_kernel_loop_iter_counter[8] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 1);
}if (private_ocl_kernel_loop_iter_counter[8] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 2);
}if (private_ocl_kernel_loop_iter_counter[8] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[8]) {
    atomic_or(&my_ocl_kernel_loop_recorder[8], 8);
}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[7] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 1);
}if (private_ocl_kernel_loop_iter_counter[7] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 2);
}if (private_ocl_kernel_loop_iter_counter[7] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[7]) {
    atomic_or(&my_ocl_kernel_loop_recorder[7], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(7,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_cols; i++){
				checksum[5] = checksum[5]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[9] = 0;
private_ocl_kernel_loop_boundary_not_reached[9] = true;
while(ei_new < d_common.in2_pad_cumv_sel_elem || (private_ocl_kernel_loop_boundary_not_reached[9] = false)){
private_ocl_kernel_loop_iter_counter[9]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[28], 1);

				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[29], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[9] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 1);
}if (private_ocl_kernel_loop_iter_counter[9] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 2);
}if (private_ocl_kernel_loop_iter_counter[9] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[9]) {
    atomic_or(&my_ocl_kernel_loop_recorder[9], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(8,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[6] = checksum[6]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[10] = 0;
private_ocl_kernel_loop_boundary_not_reached[10] = true;
while(ei_new < d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[10] = false)){
private_ocl_kernel_loop_iter_counter[10]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[30], 1);

				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[31], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[10] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[10], 1);
}if (private_ocl_kernel_loop_iter_counter[10] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[10], 2);
}if (private_ocl_kernel_loop_iter_counter[10] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[10], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[10]) {
    atomic_or(&my_ocl_kernel_loop_recorder[10], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(9,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[7] = checksum[7]+d_unique_d_in2_sub_cumh[i];
			}
		}
#endif
		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(10,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[11] = 0;
private_ocl_kernel_loop_boundary_not_reached[11] = true;
while(ei_new < d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[11] = false)){
private_ocl_kernel_loop_iter_counter[11]++;


			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[11] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[11], 1);
}if (private_ocl_kernel_loop_iter_counter[11] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[11], 2);
}if (private_ocl_kernel_loop_iter_counter[11] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[11], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[11]) {
    atomic_or(&my_ocl_kernel_loop_recorder[11], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(11,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[8] = checksum[8]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[12] = 0;
private_ocl_kernel_loop_boundary_not_reached[12] = true;
while(ei_new < d_common.in2_sub_cumh_rows || (private_ocl_kernel_loop_boundary_not_reached[12] = false)){
private_ocl_kernel_loop_iter_counter[12]++;


			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			private_ocl_kernel_loop_iter_counter[13] = 0;
private_ocl_kernel_loop_boundary_not_reached[13] = true;
for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[13] = false); position = position + d_common.in2_sub_cumh_rows){
private_ocl_kernel_loop_iter_counter[13]++;

				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}
if (private_ocl_kernel_loop_iter_counter[13] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[13], 1);
}if (private_ocl_kernel_loop_iter_counter[13] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[13], 2);
}if (private_ocl_kernel_loop_iter_counter[13] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[13], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[13]) {
    atomic_or(&my_ocl_kernel_loop_recorder[13], 8);
}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[12] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[12], 1);
}if (private_ocl_kernel_loop_iter_counter[12] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[12], 2);
}if (private_ocl_kernel_loop_iter_counter[12] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[12], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[12]) {
    atomic_or(&my_ocl_kernel_loop_recorder[12], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(12,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[9] = checksum[9]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[14] = 0;
private_ocl_kernel_loop_boundary_not_reached[14] = true;
while(ei_new < d_common.in2_sub_cumh_sel_elem || (private_ocl_kernel_loop_boundary_not_reached[14] = false)){
private_ocl_kernel_loop_iter_counter[14]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[32], 1);

				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[33], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[14] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[14], 1);
}if (private_ocl_kernel_loop_iter_counter[14] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[14], 2);
}if (private_ocl_kernel_loop_iter_counter[14] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[14], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[14]) {
    atomic_or(&my_ocl_kernel_loop_recorder[14], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(13,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[10] = checksum[10]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[15] = 0;
private_ocl_kernel_loop_boundary_not_reached[15] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[15] = false)){
private_ocl_kernel_loop_iter_counter[15]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[34], 1);

				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[35], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[15] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[15], 1);
}if (private_ocl_kernel_loop_iter_counter[15] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[15], 2);
}if (private_ocl_kernel_loop_iter_counter[15] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[15], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[15]) {
    atomic_or(&my_ocl_kernel_loop_recorder[15], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(14,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[11] = checksum[11]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[16] = 0;
private_ocl_kernel_loop_boundary_not_reached[16] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[16] = false)){
private_ocl_kernel_loop_iter_counter[16]++;


			// subtract
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[16] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[16], 1);
}if (private_ocl_kernel_loop_iter_counter[16] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[16], 2);
}if (private_ocl_kernel_loop_iter_counter[16] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[16], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[16]) {
    atomic_or(&my_ocl_kernel_loop_recorder[16], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(15,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[12] = checksum[12]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	CUMULATIVE SUM 2
		//====================================================================================================100

		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[17] = 0;
private_ocl_kernel_loop_boundary_not_reached[17] = true;
while(ei_new < d_common.in2_sqr_elem || (private_ocl_kernel_loop_boundary_not_reached[17] = false)){
private_ocl_kernel_loop_iter_counter[17]++;


			temp = d_unique_d_in2[ei_new];
			d_unique_d_in2_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[17] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[17], 1);
}if (private_ocl_kernel_loop_iter_counter[17] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[17], 2);
}if (private_ocl_kernel_loop_iter_counter[17] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[17], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[17]) {
    atomic_or(&my_ocl_kernel_loop_recorder[17], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(16,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sqr_elem; i++){
				checksum[13] = checksum[13]+d_unique_d_in2_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//==================================================50

		//==================================================50
		//	PAD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[18] = 0;
private_ocl_kernel_loop_boundary_not_reached[18] = true;
while(ei_new < d_common.in2_pad_cumv_elem || (private_ocl_kernel_loop_boundary_not_reached[18] = false)){
private_ocl_kernel_loop_iter_counter[18]++;


			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[36], 1);

				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[37], 1);
}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_sqr_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_sqr_cols)){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[38], 1);

				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2_sqr[ori_col*d_common.in2_sqr_rows+ori_row];
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[39], 1);
																							// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[18] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[18], 1);
}if (private_ocl_kernel_loop_iter_counter[18] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[18], 2);
}if (private_ocl_kernel_loop_iter_counter[18] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[18], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[18]) {
    atomic_or(&my_ocl_kernel_loop_recorder[18], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(17,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[14] = checksum[14]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[19] = 0;
private_ocl_kernel_loop_boundary_not_reached[19] = true;
while(ei_new < d_common.in2_pad_cumv_cols || (private_ocl_kernel_loop_boundary_not_reached[19] = false)){
private_ocl_kernel_loop_iter_counter[19]++;


			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			private_ocl_kernel_loop_iter_counter[20] = 0;
private_ocl_kernel_loop_boundary_not_reached[20] = true;
for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows || (private_ocl_kernel_loop_boundary_not_reached[20] = false); position = position + 1){
private_ocl_kernel_loop_iter_counter[20]++;

				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}
if (private_ocl_kernel_loop_iter_counter[20] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[20], 1);
}if (private_ocl_kernel_loop_iter_counter[20] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[20], 2);
}if (private_ocl_kernel_loop_iter_counter[20] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[20], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[20]) {
    atomic_or(&my_ocl_kernel_loop_recorder[20], 8);
}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[19] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[19], 1);
}if (private_ocl_kernel_loop_iter_counter[19] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[19], 2);
}if (private_ocl_kernel_loop_iter_counter[19] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[19], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[19]) {
    atomic_or(&my_ocl_kernel_loop_recorder[19], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(18,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[15] = checksum[15]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[21] = 0;
private_ocl_kernel_loop_boundary_not_reached[21] = true;
while(ei_new < d_common.in2_pad_cumv_sel_elem || (private_ocl_kernel_loop_boundary_not_reached[21] = false)){
private_ocl_kernel_loop_iter_counter[21]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[40], 1);

				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[41], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[21] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[21], 1);
}if (private_ocl_kernel_loop_iter_counter[21] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[21], 2);
}if (private_ocl_kernel_loop_iter_counter[21] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[21], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[21]) {
    atomic_or(&my_ocl_kernel_loop_recorder[21], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(19,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[16] = checksum[16]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[22] = 0;
private_ocl_kernel_loop_boundary_not_reached[22] = true;
while(ei_new < d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[22] = false)){
private_ocl_kernel_loop_iter_counter[22]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[42], 1);

				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[43], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[22] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[22], 1);
}if (private_ocl_kernel_loop_iter_counter[22] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[22], 2);
}if (private_ocl_kernel_loop_iter_counter[22] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[22], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[22]) {
    atomic_or(&my_ocl_kernel_loop_recorder[22], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(20,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[17] = checksum[17]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[23] = 0;
private_ocl_kernel_loop_boundary_not_reached[23] = true;
while(ei_new < d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[23] = false)){
private_ocl_kernel_loop_iter_counter[23]++;


			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[23] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[23], 1);
}if (private_ocl_kernel_loop_iter_counter[23] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[23], 2);
}if (private_ocl_kernel_loop_iter_counter[23] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[23], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[23]) {
    atomic_or(&my_ocl_kernel_loop_recorder[23], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(21,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[18] = checksum[18]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[24] = 0;
private_ocl_kernel_loop_boundary_not_reached[24] = true;
while(ei_new < d_common.in2_sub_cumh_rows || (private_ocl_kernel_loop_boundary_not_reached[24] = false)){
private_ocl_kernel_loop_iter_counter[24]++;


			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			private_ocl_kernel_loop_iter_counter[25] = 0;
private_ocl_kernel_loop_boundary_not_reached[25] = true;
for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem || (private_ocl_kernel_loop_boundary_not_reached[25] = false); position = position + d_common.in2_sub_cumh_rows){
private_ocl_kernel_loop_iter_counter[25]++;

				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}
if (private_ocl_kernel_loop_iter_counter[25] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[25], 1);
}if (private_ocl_kernel_loop_iter_counter[25] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[25], 2);
}if (private_ocl_kernel_loop_iter_counter[25] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[25], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[25]) {
    atomic_or(&my_ocl_kernel_loop_recorder[25], 8);
}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[24] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[24], 1);
}if (private_ocl_kernel_loop_iter_counter[24] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[24], 2);
}if (private_ocl_kernel_loop_iter_counter[24] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[24], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[24]) {
    atomic_or(&my_ocl_kernel_loop_recorder[24], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(22,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_rows; i++){
				checksum[19] = checksum[19]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[26] = 0;
private_ocl_kernel_loop_boundary_not_reached[26] = true;
while(ei_new < d_common.in2_sub_cumh_sel_elem || (private_ocl_kernel_loop_boundary_not_reached[26] = false)){
private_ocl_kernel_loop_iter_counter[26]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[44], 1);

				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[45], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[26] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[26], 1);
}if (private_ocl_kernel_loop_iter_counter[26] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[26], 2);
}if (private_ocl_kernel_loop_iter_counter[26] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[26], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[26]) {
    atomic_or(&my_ocl_kernel_loop_recorder[26], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(23,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[20] = checksum[20]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[27] = 0;
private_ocl_kernel_loop_boundary_not_reached[27] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[27] = false)){
private_ocl_kernel_loop_iter_counter[27]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[46], 1);

				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[47], 1);
}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[27] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[27], 1);
}if (private_ocl_kernel_loop_iter_counter[27] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[27], 2);
}if (private_ocl_kernel_loop_iter_counter[27] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[27], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[27]) {
    atomic_or(&my_ocl_kernel_loop_recorder[27], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(24,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[21] = checksum[21]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[28] = 0;
private_ocl_kernel_loop_boundary_not_reached[28] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[28] = false)){
private_ocl_kernel_loop_iter_counter[28]++;


			// subtract
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[28] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[28], 1);
}if (private_ocl_kernel_loop_iter_counter[28] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[28], 2);
}if (private_ocl_kernel_loop_iter_counter[28] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[28], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[28]) {
    atomic_or(&my_ocl_kernel_loop_recorder[28], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(25,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[22] = checksum[22]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	FINAL
		//====================================================================================================100

		//==================================================50
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[29] = 0;
private_ocl_kernel_loop_boundary_not_reached[29] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[29] = false)){
private_ocl_kernel_loop_iter_counter[29]++;


			temp = d_unique_d_in2_sub2[ei_new];
			temp2 = d_unique_d_in2_sqr_sub2[ei_new] - (temp * temp / d_common.in_elem);
			if(temp2 < 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[48], 1);

				temp2 = 0;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[49], 1);
}
			d_unique_d_in2_sqr_sub2[ei_new] = sqrt(temp2);
			

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[29] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[29], 1);
}if (private_ocl_kernel_loop_iter_counter[29] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[29], 2);
}if (private_ocl_kernel_loop_iter_counter[29] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[29], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[29]) {
    atomic_or(&my_ocl_kernel_loop_recorder[29], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(26,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[23] = checksum[23]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[30] = 0;
private_ocl_kernel_loop_boundary_not_reached[30] = true;
while(ei_new < d_common.in_sqr_elem || (private_ocl_kernel_loop_boundary_not_reached[30] = false)){
private_ocl_kernel_loop_iter_counter[30]++;


			temp = d_in[ei_new];
			d_unique_d_in_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[30] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[30], 1);
}if (private_ocl_kernel_loop_iter_counter[30] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[30], 2);
}if (private_ocl_kernel_loop_iter_counter[30] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[30], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[30]) {
    atomic_or(&my_ocl_kernel_loop_recorder[30], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(27,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_elem; i++){
				checksum[24] = checksum[24]+d_unique_d_in_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	IN SUM
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[31] = 0;
private_ocl_kernel_loop_boundary_not_reached[31] = true;
while(ei_new < d_common.in_cols || (private_ocl_kernel_loop_boundary_not_reached[31] = false)){
private_ocl_kernel_loop_iter_counter[31]++;


			sum = 0;
			private_ocl_kernel_loop_iter_counter[32] = 0;
private_ocl_kernel_loop_boundary_not_reached[32] = true;
for(i = 0; i < d_common.in_rows || (private_ocl_kernel_loop_boundary_not_reached[32] = false); i++){
private_ocl_kernel_loop_iter_counter[32]++;


				sum = sum + d_in[ei_new*d_common.in_rows+i];

			}
if (private_ocl_kernel_loop_iter_counter[32] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[32], 1);
}if (private_ocl_kernel_loop_iter_counter[32] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[32], 2);
}if (private_ocl_kernel_loop_iter_counter[32] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[32], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[32]) {
    atomic_or(&my_ocl_kernel_loop_recorder[32], 8);
}
			in_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[31] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[31], 1);
}if (private_ocl_kernel_loop_iter_counter[31] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[31], 2);
}if (private_ocl_kernel_loop_iter_counter[31] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[31], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[31]) {
    atomic_or(&my_ocl_kernel_loop_recorder[31], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(28,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_cols; i++){
				checksum[25] = checksum[25]+in_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	IN_SQR SUM
		//==================================================50

		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[33] = 0;
private_ocl_kernel_loop_boundary_not_reached[33] = true;
while(ei_new < d_common.in_sqr_rows || (private_ocl_kernel_loop_boundary_not_reached[33] = false)){
private_ocl_kernel_loop_iter_counter[33]++;

				
			sum = 0;
			private_ocl_kernel_loop_iter_counter[34] = 0;
private_ocl_kernel_loop_boundary_not_reached[34] = true;
for(i = 0; i < d_common.in_sqr_cols || (private_ocl_kernel_loop_boundary_not_reached[34] = false); i++){
private_ocl_kernel_loop_iter_counter[34]++;


				sum = sum + d_unique_d_in_sqr[ei_new+d_common.in_sqr_rows*i];

			}
if (private_ocl_kernel_loop_iter_counter[34] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[34], 1);
}if (private_ocl_kernel_loop_iter_counter[34] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[34], 2);
}if (private_ocl_kernel_loop_iter_counter[34] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[34], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[34]) {
    atomic_or(&my_ocl_kernel_loop_recorder[34], 8);
}
			in_sqr_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[33] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[33], 1);
}if (private_ocl_kernel_loop_iter_counter[33] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[33], 2);
}if (private_ocl_kernel_loop_iter_counter[33] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[33], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[33]) {
    atomic_or(&my_ocl_kernel_loop_recorder[33], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(29,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_rows; i++){
				checksum[26] = checksum[26]+in_sqr_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	FINAL SUMMATION
		//==================================================50

		if(tx == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[50], 1);


			in_final_sum[0] = 0;
			private_ocl_kernel_loop_iter_counter[35] = 0;
private_ocl_kernel_loop_boundary_not_reached[35] = true;
for(i = 0; i<d_common.in_cols || (private_ocl_kernel_loop_boundary_not_reached[35] = false); i++){
private_ocl_kernel_loop_iter_counter[35]++;

				// in_final_sum = in_final_sum + in_partial_sum[i];
				in_final_sum[0] = in_final_sum[0] + in_partial_sum[i];
			}
if (private_ocl_kernel_loop_iter_counter[35] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[35], 1);
}if (private_ocl_kernel_loop_iter_counter[35] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[35], 2);
}if (private_ocl_kernel_loop_iter_counter[35] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[35], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[35]) {
    atomic_or(&my_ocl_kernel_loop_recorder[35], 8);
}

		}else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[51], 1);

if(tx == 1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[52], 1);


			in_sqr_final_sum[0] = 0;
			private_ocl_kernel_loop_iter_counter[36] = 0;
private_ocl_kernel_loop_boundary_not_reached[36] = true;
for(i = 0; i<d_common.in_sqr_cols || (private_ocl_kernel_loop_boundary_not_reached[36] = false); i++){
private_ocl_kernel_loop_iter_counter[36]++;

				// in_sqr_final_sum = in_sqr_final_sum + in_sqr_partial_sum[i];
				in_sqr_final_sum[0] = in_sqr_final_sum[0] + in_sqr_partial_sum[i];
			}
if (private_ocl_kernel_loop_iter_counter[36] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[36], 1);
}if (private_ocl_kernel_loop_iter_counter[36] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[36], 2);
}if (private_ocl_kernel_loop_iter_counter[36] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[36], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[36]) {
    atomic_or(&my_ocl_kernel_loop_recorder[36], 8);
}

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[53], 1);
}
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(30,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[27] = checksum[27]+in_final_sum[0]+in_sqr_final_sum[0];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	DENOMINATOR T
		//==================================================50

		if(tx == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[54], 1);


			// mean = in_final_sum / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean = in_final_sum[0] / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			// variance  = (in_sqr_final_sum / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			variance  = (in_sqr_final_sum[0] / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			// denomT = sqrt((float)(d_common.in_elem-1))*deviation;
			denomT[0] = sqrt((float)(d_common.in_elem-1))*deviation;

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[55], 1);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(31,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[28] = checksum[28]+denomT[i];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[37] = 0;
private_ocl_kernel_loop_boundary_not_reached[37] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[37] = false)){
private_ocl_kernel_loop_iter_counter[37]++;


			// d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT[0];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[37] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[37], 1);
}if (private_ocl_kernel_loop_iter_counter[37] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[37], 2);
}if (private_ocl_kernel_loop_iter_counter[37] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[37], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[37]) {
    atomic_or(&my_ocl_kernel_loop_recorder[37], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(32,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[29] = checksum[29]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[38] = 0;
private_ocl_kernel_loop_boundary_not_reached[38] = true;
while(ei_new < d_common.conv_elem || (private_ocl_kernel_loop_boundary_not_reached[38] = false)){
private_ocl_kernel_loop_iter_counter[38]++;


			// d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;
			d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum[0] / d_common.in_elem;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[38] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[38], 1);
}if (private_ocl_kernel_loop_iter_counter[38] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[38], 2);
}if (private_ocl_kernel_loop_iter_counter[38] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[38], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[38]) {
    atomic_or(&my_ocl_kernel_loop_recorder[38], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(33,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[30] = checksum[30]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[39] = 0;
private_ocl_kernel_loop_boundary_not_reached[39] = true;
while(ei_new < d_common.in2_sub2_elem || (private_ocl_kernel_loop_boundary_not_reached[39] = false)){
private_ocl_kernel_loop_iter_counter[39]++;


			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_conv[ei_new] / d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[39] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[39], 1);
}if (private_ocl_kernel_loop_iter_counter[39] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[39], 2);
}if (private_ocl_kernel_loop_iter_counter[39] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[39], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[39]) {
    atomic_or(&my_ocl_kernel_loop_recorder[39], 8);
}



		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(34,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[31] = checksum[31]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	TEMPLATE MASK CREATE
		//====================================================================================================100

		cent = d_common.sSize + d_common.tSize + 1;
		if(d_frame_no == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[56], 1);

			tMask_row = cent + d_unique_d_Row[d_unique_point_no] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_Col[d_unique_point_no] - d_unique_d_Col[d_unique_point_no] - 1;
		}
		else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[57], 1);

			pointer = d_unique_point_no*d_common.no_frames+d_frame_no-1;
			tMask_row = cent + d_unique_d_tRowLoc[pointer] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_tColLoc[pointer] - d_unique_d_Col[d_unique_point_no] - 1;
		}

		//work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[40] = 0;
private_ocl_kernel_loop_boundary_not_reached[40] = true;
while(ei_new < d_common.tMask_elem || (private_ocl_kernel_loop_boundary_not_reached[40] = false)){
private_ocl_kernel_loop_iter_counter[40]++;


			location = tMask_col*d_common.tMask_rows + tMask_row;

			if(ei_new==location){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[58], 1);

				d_unique_d_tMask[ei_new] = 1;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[59], 1);

				d_unique_d_tMask[ei_new] = 0;
			}

			//go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[40] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[40], 1);
}if (private_ocl_kernel_loop_iter_counter[40] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[40], 2);
}if (private_ocl_kernel_loop_iter_counter[40] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[40], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[40]) {
    atomic_or(&my_ocl_kernel_loop_recorder[40], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(35,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.tMask_elem; i++){
				checksum[32] = checksum[32]+d_unique_d_tMask[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MASK CONVOLUTION
		//====================================================================================================100

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[41] = 0;
private_ocl_kernel_loop_boundary_not_reached[41] = true;
while(ei_new < d_common.mask_conv_elem || (private_ocl_kernel_loop_boundary_not_reached[41] = false)){
private_ocl_kernel_loop_iter_counter[41]++;


			// figure out row/col location in array
			ic = (ei_new+1) % d_common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.mask_conv_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[60], 1);

				ic = d_common.mask_conv_rows;
				jc = jc-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[61], 1);
}

			//
			j = jc + d_common.mask_conv_joffset;
			jp1 = j + 1;
			if(d_common.mask_cols < jp1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[62], 1);

				ja1 = jp1 - d_common.mask_cols;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[63], 1);

				ja1 = 1;
			}
			if(d_common.tMask_cols < j){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[64], 1);

				ja2 = d_common.tMask_cols;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[65], 1);

				ja2 = j;
			}

			i = ic + d_common.mask_conv_ioffset;
			ip1 = i + 1;
			
			if(d_common.mask_rows < ip1){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[66], 1);

				ia1 = ip1 - d_common.mask_rows;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[67], 1);

				ia1 = 1;
			}
			if(d_common.tMask_rows < i){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[68], 1);

				ia2 = d_common.tMask_rows;
			}
			else{
atomic_or(&my_ocl_kernel_branch_triggered_recorder[69], 1);

				ia2 = i;
			}

			s = 0;

			private_ocl_kernel_loop_iter_counter[42] = 0;
private_ocl_kernel_loop_boundary_not_reached[42] = true;
for(ja=ja1; ja<=ja2 || (private_ocl_kernel_loop_boundary_not_reached[42] = false); ja++){
private_ocl_kernel_loop_iter_counter[42]++;

				jb = jp1 - ja;
				private_ocl_kernel_loop_iter_counter[43] = 0;
private_ocl_kernel_loop_boundary_not_reached[43] = true;
for(ia=ia1; ia<=ia2 || (private_ocl_kernel_loop_boundary_not_reached[43] = false); ia++){
private_ocl_kernel_loop_iter_counter[43]++;

					ib = ip1 - ia;
					s = s + d_unique_d_tMask[d_common.tMask_rows*(ja-1)+ia-1] * 1;
				}
if (private_ocl_kernel_loop_iter_counter[43] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[43], 1);
}if (private_ocl_kernel_loop_iter_counter[43] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[43], 2);
}if (private_ocl_kernel_loop_iter_counter[43] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[43], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[43]) {
    atomic_or(&my_ocl_kernel_loop_recorder[43], 8);
}
			}
if (private_ocl_kernel_loop_iter_counter[42] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[42], 1);
}if (private_ocl_kernel_loop_iter_counter[42] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[42], 2);
}if (private_ocl_kernel_loop_iter_counter[42] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[42], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[42]) {
    atomic_or(&my_ocl_kernel_loop_recorder[42], 8);
}

			// //d_unique_d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_mask_conv[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[41] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[41], 1);
}if (private_ocl_kernel_loop_iter_counter[41] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[41], 2);
}if (private_ocl_kernel_loop_iter_counter[41] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[41], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[41]) {
    atomic_or(&my_ocl_kernel_loop_recorder[41], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(36,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_elem; i++){
				checksum[33] = checksum[33]+d_unique_d_mask_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MAXIMUM VALUE
		//====================================================================================================100

		//==================================================50
		//	INITIAL SEARCH
		//==================================================50

		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[44] = 0;
private_ocl_kernel_loop_boundary_not_reached[44] = true;
while(ei_new < d_common.mask_conv_rows || (private_ocl_kernel_loop_boundary_not_reached[44] = false)){
private_ocl_kernel_loop_iter_counter[44]++;


			private_ocl_kernel_loop_iter_counter[45] = 0;
private_ocl_kernel_loop_boundary_not_reached[45] = true;
for(i=0; i<d_common.mask_conv_cols || (private_ocl_kernel_loop_boundary_not_reached[45] = false); i++){
private_ocl_kernel_loop_iter_counter[45]++;

				largest_coordinate_current = ei_new*d_common.mask_conv_rows+i;
				largest_value_current = fabs(d_unique_d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[70], 1);

					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[71], 1);
}
			}
if (private_ocl_kernel_loop_iter_counter[45] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[45], 1);
}if (private_ocl_kernel_loop_iter_counter[45] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[45], 2);
}if (private_ocl_kernel_loop_iter_counter[45] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[45], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[45]) {
    atomic_or(&my_ocl_kernel_loop_recorder[45], 8);
}
			par_max_coo[ei_new] = largest_coordinate;
			par_max_val[ei_new] = largest_value;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[44] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[44], 1);
}if (private_ocl_kernel_loop_iter_counter[44] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[44], 2);
}if (private_ocl_kernel_loop_iter_counter[44] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[44], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[44]) {
    atomic_or(&my_ocl_kernel_loop_recorder[44], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(37,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_rows; i++){
				checksum[34] = checksum[34]+par_max_coo[i]+par_max_val[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	FINAL SEARCH
		//==================================================50

		if(tx == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[72], 1);


			private_ocl_kernel_loop_iter_counter[46] = 0;
private_ocl_kernel_loop_boundary_not_reached[46] = true;
for(i = 0; i < d_common.mask_conv_rows || (private_ocl_kernel_loop_boundary_not_reached[46] = false); i++){
private_ocl_kernel_loop_iter_counter[46]++;

				if(par_max_val[i] > fin_max_val){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[74], 1);

					fin_max_val = par_max_val[i];
					fin_max_coo = par_max_coo[i];
				}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[75], 1);
}
			}
if (private_ocl_kernel_loop_iter_counter[46] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[46], 1);
}if (private_ocl_kernel_loop_iter_counter[46] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[46], 2);
}if (private_ocl_kernel_loop_iter_counter[46] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[46], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[46]) {
    atomic_or(&my_ocl_kernel_loop_recorder[46], 8);
}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % d_common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / d_common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % d_common.mask_conv_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[76], 1);

				largest_row = d_common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[77], 1);
}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
			offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
			pointer = d_unique_point_no*d_common.no_frames+d_frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no] + offset_row;
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no] + offset_col;

		}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[73], 1);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(38,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[35] = checksum[35]+d_unique_d_tRowLoc[pointer]+d_unique_d_tColLoc[pointer];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[9], 1);
}

	//======================================================================================================================================================150
	//	PERIODIC COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	if(d_frame_no != 0 && (d_frame_no)%10 == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[78], 1);


		//====================================================================================================100
		// initialize cross-frame variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block
			for(i=0; i<d_common.in_elem; i++){
				d_in[i] = 0;
			}

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		OCL_NEW_BARRIER(39,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		// if the last frame in the bath, update template
		//====================================================================================================100

		// update coordinate
		loc_pointer = d_unique_point_no*d_common.no_frames+d_frame_no;

		d_unique_d_Row[d_unique_point_no] = d_unique_d_tRowLoc[loc_pointer];
		d_unique_d_Col[d_unique_point_no] = d_unique_d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		private_ocl_kernel_loop_iter_counter[47] = 0;
private_ocl_kernel_loop_boundary_not_reached[47] = true;
while(ei_new < d_common.in_elem || (private_ocl_kernel_loop_boundary_not_reached[47] = false)){
private_ocl_kernel_loop_iter_counter[47]++;


			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
atomic_or(&my_ocl_kernel_branch_triggered_recorder[80], 1);

				row = d_common.in_rows - 1;
				col = col-1;
			}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[81], 1);
}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = d_common.alpha*d_in[ei_new] + (1-d_common.alpha)*d_common_change_d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}
if (private_ocl_kernel_loop_iter_counter[47] == 0) {
    atomic_or(&my_ocl_kernel_loop_recorder[47], 1);
}if (private_ocl_kernel_loop_iter_counter[47] == 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[47], 2);
}if (private_ocl_kernel_loop_iter_counter[47] > 1) {
    atomic_or(&my_ocl_kernel_loop_recorder[47], 4);
}if (!private_ocl_kernel_loop_boundary_not_reached[47]) {
    atomic_or(&my_ocl_kernel_loop_recorder[47], 8);
}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		OCL_NEW_BARRIER(40,CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[36] = checksum[36]+d_in[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}
else {

atomic_or(&my_ocl_kernel_branch_triggered_recorder[79], 1);
}

	//======================================================================================================================================================150
	//	End
	//======================================================================================================================================================150

for (int update_recorder_i = 0; update_recorder_i < 82; update_recorder_i++) { 
  atomic_or(&ocl_kernel_branch_triggered_recorder[update_recorder_i], my_ocl_kernel_branch_triggered_recorder[update_recorder_i]); 
}
for (int update_recorder_i = 0; update_recorder_i < 48; update_recorder_i++) { 
  atomic_or(&ocl_kernel_loop_recorder[update_recorder_i], my_ocl_kernel_loop_recorder[update_recorder_i]); 
}
}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
