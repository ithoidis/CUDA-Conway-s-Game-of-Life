%   Iordanis P. Thoidis
%   Student @ AUTH
%

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define NSTEPS 1000
#define THRESHOLD 0.4

struct timeval startwtime, endwtime;
double seq_time;

// Set in A a sub of B
__device__ void SetXsub( int *A, int *B, int startx, int starty, int endx, int endy, int N){

  int i, j;
  int N1 = endx - startx;
  int N2 = endy - starty;
  for (i = 0; i <= N1+1; i++){ 
    for (j = 0; i <= N2+1; j++){
      A[i*N1 + j] = B[(startx-1+i)*N + j + starty-1]; 
    }   
  }
}

extern __shared__ float array[];
// Kernel processing multiple cells per thread with shared memory
__global__ void lifeKernelc(int *X, int *newX, int N){
    int i, j;

    int N1, cellstartx, cellstarty, cellendx, cellendy;


    int threadnum = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int threadnumx = blockDim.x * gridDim.x;
    int threadnumy = blockDim.y * gridDim.y;

    if ( threadIdx.x==threadnumx-1 && threadIdx.y==threadnumy-1){

    int cellssparedx = N % threadnumx;
    int cellssparedy = N % threadnumy;

    N1 = cellssparedx;

    cellstartx = N - cellssparedx + 1;
    cellstarty = N - cellssparedy + 1;

    cellendx = N;
    cellendy = N; 

    } else {

    int cellsperthread = (N * N) / threadnum;
    int cellsperthreadx = N / threadnumx;
    int cellsperthready = N / threadnumy;
    
    N1 = cellsperthread;
    
    cellstartx = 1 + (blockIdx.x * blockDim.x + threadIdx.x) * cellsperthread;
    cellstarty = 1 + (blockIdx.y * blockDim.y + threadIdx.y) * cellsperthread;

    cellendx = cellstartx + cellsperthreadx;
    cellendy = cellstarty + cellsperthready;
    }

    int size=(cellendx - cellstartx + 2)*(cellendy - cellstarty + 2);


    short* array0 = (short*)array; 
    int* Xsub = (int*)&array0[size];

    SetXsub(Xsub, X, cellstartx, cellstarty, cellendx, cellendy, N);


   for(i = 1; i <= cellendx - cellstartx; i++) {
     for ( j = 1; j <= cellendy - cellstarty; j++){

      //  if (i > 0 && i <= N && j > 0 && j <= N ) {

        int nsum;
        int im = i-1;
        int ip = i+1;
        int jm = j-1;
        int jp = j+1;

        nsum =  Xsub[im*N1 + jm] + Xsub[i*N1 + jm] + Xsub[ip*N1 + jm]
              + Xsub[im*N1 + j]                    + Xsub[ip*N1 + j] 
              + Xsub[im*N1 + jp] + Xsub[i*N1 + jp] + Xsub[ip*N1 + jp];
        

        int i1 = i + cellstartx;
        int j1 = i + cellstarty;

        switch(nsum){
          case 3:
            newX[i1*N + j1] = 1;
            break;

          case 2:
            newX[i1*N + j1] = Xsub[i*N1 + j];
            break;

          default:
             newX[i1*N + j1] = 0;
        }

        /* left-right boundary conditions */    
        if (j1 == 1)
          newX[i1*N + N+1] = Xsub[i*N1 + j]; // newX[i1*N + j];
        if (j1 == N)
          newX[i1*N + 0] = Xsub[i*N1 + j]; // newX[i1*N + N];
        /* top-bottom boundary conditions */
        if (i1 == 1)
          newX[(N+1)*N + j1] = Xsub[i*N1 + j]; // newX[1*N + j];
        if (i1 == N)
          newX[0*N + j1] = Xsub[i*N1 + j]; // newX[N*N + j];
     
        /* corner boundary conditions */
        if (i1==N && j1==N)
          newX[0*N + 0] = Xsub[i*N1 + j]; // newX[N*N + N];
        if (i1==N && j1==1)
          newX[0*N + N+1] = Xsub[i*N1 + j]; // newX[N*N + 1];
        if (i1==1 && j1==1)
          newX[(N+1)*N + N+1] = Xsub[i*N1 + j]; // newX[1*N + 1];
        if (i1==1 && j1==N)
          newX[(N+1)*N + 0] = Xsub[i*N1 + j]; // newX[1*N + N];
       
       }
     }
}

// Host Code to multiple cells per thread with shared memory
void process_c(int *X, int N){

    int i, n;
    int *newX = (int *) calloc(N+2,(N+2)*sizeof(int));
    size_t size = (N+2) * (N+2) * sizeof(int);
   
    // Allocate X in device memory
    int *d_X;
    cudaMalloc(&d_X, size);

    // Allocate newX in device memory
    int *d_newX;
    cudaMalloc(&d_newX, size);

    /* corner boundary conditions */
    X[0*N + 0] = X[N*N + N];
    X[0*N + N+1] = X[N*N + 1] ;
    X[(N+1)*N + N+1] = X[1*N + 1];
    X[(N+1)*N + 0] = X[1*N + N];

    for(i=1; i<=N; i++){
     /* left-right boundary conditions */      
      X[i*N + 0] = X[i*N + N];
      X[i*N + N+1] = X[i*N + 1];
     /* top-bottom boundary conditions */
      X[0*N + i] = X[N*N + i];
      X[(N+1)*N + i] = X[1*N + i]; 
    }

     // time steps 
    for(n=0; n<NSTEPS; n++){

      // Load X to device memory
      cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);

      // Kernel invocation
      dim3 threadsPerBlock(16, 16);
      dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
      
      lifeKernelc<<<numBlocks, threadsPerBlock>>>(d_X, d_newX, N);

      // Read newX from device memory
      cudaMemcpy(X, d_newX, size,
               cudaMemcpyDeviceToHost);
      }

      // Free device memory
      cudaFree(d_X);
      cudaFree(d_newX);
}

//Kernel processing multiple cells per thread with no shared memory
__global__ void lifeKernelb(int *X, int *newX, int N){

    int i, j;
    int cellstartx, cellstarty, cellendx, cellendy;

    int threadnum = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int threadnumx = blockDim.x * gridDim.x;
    int threadnumy = blockDim.y * gridDim.y;

    if ( threadIdx.x==threadnumx-1 && threadIdx.y==threadnumy-1){

    int cellssparedx = N % threadnumx;
    int cellssparedy = N % threadnumy;

    cellstartx = N - cellssparedx + 1;
    cellstarty = N - cellssparedy + 1;

    cellendx = N;
    cellendy = N; 

    } else {

    int cellsperthread = (N * N) / threadnum;
    int cellsperthreadx = N / threadnumx;
    int cellsperthready = N / threadnumy;

    cellstartx = 1 + (blockIdx.x * blockDim.x + threadIdx.x) * cellsperthread;
    cellstarty = 1 + (blockIdx.y * blockDim.y + threadIdx.y) * cellsperthread;

    cellendx = cellstartx + cellsperthreadx;
    cellendy = cellstarty + cellsperthready;
    }
    for (i = cellstartx; i <= cellendx; i++) {
     for ( j = cellstarty; j <= cellendy; j++){
     
        if (i > 0 && i <= N && j > 0 && j <= N ) {
        
        int nsum;
        int im = i-1;
        int ip = i+1;
        int jm = j-1;
        int jp = j+1;

        nsum =  X[im*N + jm] + X[i*N + jm] + X[ip*N + jm]
              + X[im*N + j]                + X[ip*N + j] 
              + X[im*N + jp] + X[i*N + jp] + X[ip*N + jp];
            
        switch(nsum){
          case 3:
            newX[i*N + j] = 1;
            break;

          case 2:
            newX[i*N + j] = X[i*N + j];
            break;

          default:
             newX[i*N + j] = 0;
        }

        /* left-right boundary conditions */    
        if (j == 1)
          newX[i*N + N+1] = newX[i*N + 1];
        if (j == N)
          newX[i*N + 0] = newX[i*N + N];
        /* top-bottom boundary conditions */
        if (i == 1)
          newX[(N+1)*N + j] = newX[1*N + j];
        if (i == N)
          newX[0*N + j] = newX[N*N + j];
     
        /* corner boundary conditions */
        if (i==N && j==N)
          newX[0*N + 0] = newX[N*N + N];
        if (i==N && j==1)
          newX[0*N + N+1] = newX[N*N + 1];
        if (i==1 && j==1)
          newX[(N+1)*N + N+1] = newX[1*N + 1];
        if (i==1 && j==N)
          newX[(N+1)*N + 0] = newX[1*N + N];
       }
      }
    }
}

// Host Code to multiple cells per thread with no shared memory
void process_b(int *X, int N){
    int i, n;
    int *newX = (int *) calloc(N+2,(N+2)*sizeof(int));
    size_t size = (N+2) * (N+2) * sizeof(int);
   
    // Allocate X in device memory
    int *d_X;
    cudaMalloc(&d_X, size);

    // Allocate newX in device memory
    int *d_newX;
    cudaMalloc(&d_newX, size);

    /* corner boundary conditions */
    X[0*N + 0] = X[N*N + N];
    X[0*N + N+1] = X[N*N + 1] ;
    X[(N+1)*N + N+1] = X[1*N + 1];
    X[(N+1)*N + 0] = X[1*N + N];

    for(i=1; i<=N; i++){
     /* left-right boundary conditions */      
      X[i*N + 0] = X[i*N + N];
      X[i*N + N+1] = X[i*N + 1];
     /* top-bottom boundary conditions */
      X[0*N + i] = X[N*N + i];
      X[(N+1)*N + i] = X[1*N + i]; 
    }

     // time steps 
    for(n=0; n<NSTEPS; n++){

      // Load X to device memory
      cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
   


      // Kernel invocation
      dim3 threadsPerBlock(16, 16);
      dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
      lifeKernelb<<<numBlocks, threadsPerBlock>>>(d_X, d_newX, N);

      // Read newX from device memory
      cudaMemcpy(X, d_newX, size, cudaMemcpyDeviceToHost);
      }

      // Free device memory
      cudaFree(d_X);
      cudaFree(&d_newX);
}


// Kernel processing one cell per thread with no shared memory
__global__ void lifeKernela(int *X, int *newX, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= N && j > 0 && j <= N ) {
        
        int im = i-1;
        int ip = i+1;
        int jm = j-1;
        int jp = j+1;
        int nsum; 
        nsum =  X[im*N + jm] + X[i*N + jm] + X[ip*N + jm]
              + X[im*N + j]                + X[ip*N + j] 
              + X[im*N + jp] + X[i*N + jp] + X[ip*N + jp];
            
        switch(nsum){
          case 3:
            newX[i*N + j] = 1;
            break;

          case 2:
            newX[i*N + j] = X[i*N + j];
            break;

          default:
             newX[i*N + j] = 0;
        }

        /* left-right boundary conditions */    
        if (j == 1)
          newX[i*N + N+1] = newX[i*N + 1];
        if (j == N)
          newX[i*N + 0] = newX[i*N + N];
        /* top-bottom boundary conditions */
        if (i == 1)
          newX[(N+1)*N + j] = newX[1*N + j];
        if (i == N)
          newX[0*N + j] = newX[N*N + j];
     
        /* corner boundary conditions */
        if (i==N && j==N)
          newX[0*N + 0] = newX[N*N + N];
        if (i==N && j==1)
          newX[0*N + N+1] = newX[N*N + 1];
        if (i==1 && j==1)
          newX[(N+1)*N + N+1] = newX[1*N + 1];
        if (i==1 && j==N)
          newX[(N+1)*N + 0] = newX[1*N + N];

    }
}

// Host Code
void process_a(int *X, int N){
    int i, n;
    int *newX = (int *) calloc(N+2,(N+2)*sizeof(int));
    size_t size = (N+2) * (N+2) * sizeof(int);
   
    // Allocate X in device memory
    int* d_X;
    cudaMalloc(&d_X, size);

    // Allocate newX in device memory
    int* d_newX;
    cudaMalloc(&d_newX, size);

    /* corner boundary conditions */
    X[0*N + 0] = X[N*N + N];
    X[0*N + N+1] = X[N*N + 1] ;
    X[(N+1)*N + N+1] = X[1*N + 1];
    X[(N+1)*N + 0] = X[1*N + N];

    for(i=1; i<=N; i++){
     /* left-right boundary conditions */      
      X[i*N + 0] = X[i*N + N];
      X[i*N + N+1] = X[i*N + 1];
     /* top-bottom boundary conditions */
      X[0*N + i] = X[N*N + i];
      X[(N+1)*N + i] = X[1*N + i]; 
    }

     // time steps 
    for(n=0; n<NSTEPS; n++){

      // Load X to device memory
      cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
   
      // Kernel invocation
      dim3 threadsPerBlock(16, 16);
      int blockx = (N+2) / threadsPerBlock.x + 1;
      int blocky = (N+2) / threadsPerBlock.y + 1;
      dim3 numBlocks(blockx, blocky);
    
      lifeKernela<<<numBlocks, threadsPerBlock>>>(d_X, d_newX, N);

      // Read newX from device memory
      cudaMemcpy(X, d_newX, size, cudaMemcpyDeviceToHost);
      }

      // Free device memory
      cudaFree(d_X);
      cudaFree(d_newX);
}

void process_seq(int *X, int N){
  
  int n, i, j, im, ip, jm, jp, nsum;

  int *newX = (int *)calloc(N+2,(N+2)*sizeof(int));
 
  /*  time steps */
  for(n=0; n<NSTEPS; n++){

    /* corner boundary conditions */
    X[0*N + 0] = X[N*N + N];
    X[0*N + N+1] = X[N*N + 1] ;
    X[(N+1)*N + N+1] = X[1*N + 1];
    X[(N+1)*N + 0] = X[1*N + N];


    for(i=1; i<=N; i++){
     /* left-right boundary conditions */      
      X[i*N + 0] = X[i*N + N];
      X[i*N + N+1] = X[i*N + 1];
     /* top-bottom boundary conditions */
      X[0*N + i] = X[N*N + i];
      X[(N+1)*N + i] = X[1*N + i]; 
    }


    for(i=1; i<=N; i++){
      for(j=1; j<=N; j++){
  
  im = i-1;
  ip = i+1;
  jm = j-1;
  jp = j+1;

  nsum =  X[im*N + jp] + X[i*N + jp] + X[ip*N + jp]
        + X[im*N + j]                + X[ip*N + j] 
        + X[im*N + jm] + X[i*N + jm] + X[ip*N + jm];

  switch(nsum){

   case 3:
    newX[i*N + j] = 1;
    break;

   case 2:
    newX[i*N + j] = X[i*N + j];
    break;

   default:
    newX[i*N + j] = 0;
    
  }
      }
    }

    /* copy new state into old state */

    for(i=1; i<=N; i++){
      for(j=1; j<=N; j++){
         X[i*N + j] = newX[i*N + j];
      }
    }
  }


  free(newX);
}

void generate_table(int *X, int N){

  printf("Generating an %d x %d table\n", N, N);

  srand(time(NULL));
  int counter = 0;
  int i, j;
  for(i=1; i<=N; i++){
    for(j=1; j<=N; j++){
      X[i*N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD; 
      counter += X[i*N + j];
    }
  }

  printf("Number of non zero elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(N*N));
}

void check_table(const int *X, int N){

  int counter = 0;
  int i, j;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      counter += X[i*N + j];
    }
  }
  printf("Number of non zero elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(N*N));
}

void read_table( int *X, int N){

  char filename[20];

  sprintf(filename, "table%dx%d.bin", N, N);

  printf("Reading %dx%d table from file %s\n", N, N, filename);

  FILE *fp = fopen(filename, "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  fclose(fp);
  
  int i, j;

  // X[0..N-1] -> X[1..N], adding aux cells
  for(i=N-1; i>=0; i--){
    for(j=N-1; j>=0; j--){
      X[(i+1)*N + j+1]=  X[i*N + j]; 
    }
  }

  /*  int counter;
  for(int i=0; i<N; i++){
    printf(":\n");
    for(int j=0; j<N; j++){
       printf(" %d", X[i*N + j]);
       counter += X[i*N + j];
    }
  }
  printf("Number of non zero elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(N*N));
  */
  fclose(fp);
}

void save_table(int *X, int N){

  // X[0..N+1] -> X[0..N-1], reducing aux cells
  int i, j, offset=0;
  for(i=0; i<N; i++){
    offset++;
    for(j=0; j<N; j++){
      X[i*N + j] = X[(i+1)*N + j + offset];
    }
    offset++;
  }

  FILE *fp;

  char filename[20];

  sprintf(filename, "table%dx%d.bin", N, N);

  printf("Saving table in file %s\n", filename);

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);
}

int main(int argc, char **argv){

  int N = atoi(argv[1]);
  
  int *table = (int *)calloc((N+2),(N+2)*sizeof(int));


  //int *table = (int *)malloc(N*N*sizeof(int));
  //read_table(table, N);
  generate_table(table, N);
  gettimeofday (&startwtime, NULL);
  process_seq(table, N);
  gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  printf("Sequential Process Time = %f\n", seq_time);
  
 generate_table(table, N);
 gettimeofday (&startwtime, NULL);
 process_a(table, N);
 gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

 printf("Cuda Process Time  A    = %f\n", seq_time);

 generate_table(table, N);
 gettimeofday (&startwtime, NULL);
 process_b(table, N);
 gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

 printf("Cuda Process Time  B    = %f\n", seq_time);

  generate_table(table, N);
  gettimeofday (&startwtime, NULL);
  process_c(table, N);
  gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  printf("Cuda Process Time  C    = %f\n", seq_time);


  //check_table(table, N);

  save_table(table, N);
  
  free(table);
 
}
