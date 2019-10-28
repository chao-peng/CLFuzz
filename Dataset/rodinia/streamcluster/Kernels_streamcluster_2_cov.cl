
typedef struct {
  float weight;
  long assign;
  float cost;
} Point_Struct;


__kernel void pgain_kernel(
			 __global Point_Struct *p,
			 __global float *coord_d,
			 __global float * work_mem_d,
			 __global int *center_table_d,
			 __global char *switch_membership_d,
			 __local float *coord_s,
			 int num,
			 int dim,
			 long x,
			 int K){
	const int thread_id = get_global_id(0);
	const int local_id = get_local_id(0);

	if(thread_id<num){

	  if(local_id == 0)
	   	for(int i=0; i<dim; i++){
	   		coord_s[i] = coord_d[i*num + x];
	   	}
	  barrier(CLK_LOCAL_MEM_FENCE);

	  float x_cost = 0.0;
	  for(int i=0; i<dim; i++)
		  x_cost += (coord_d[(i*num)+thread_id]-coord_s[i]) * (coord_d[(i*num)+thread_id]-coord_s[i]);
	  x_cost = x_cost * p[thread_id].weight;

	  float current_cost = p[thread_id].cost;

	  int base = thread_id*(K+1);

	  if ( x_cost < current_cost ){
		  switch_membership_d[thread_id] = '1';
	      int addr_1 = base + K;
	      work_mem_d[addr_1] = x_cost - current_cost;
	  }

	  else {
	      int assign = p[thread_id].assign;
	      int addr_2 = base + center_table_d[assign];
	      work_mem_d[addr_2] += current_cost - x_cost;
	  }
	}
}
