
#include <stdio.h>
#include <time.h>
#include <math.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
  /* 3-lag linear operator on log bursts, online recursive least squares (forgetting 0.995) */
  double A[3]={0.5,0.2,0.1}, P[3][3]={{100,0,0},{0,100,0},{0,0,100}}, x[3]={0.1,0.2,0.3}, lam=0.995;
  volatile double sink=0; long N=10000000; double t0=now();
  for(long i=0;i<N;i++){ double y=A[0]*x[0]+A[1]*x[1]+A[2]*x[2]; sink+=y; x[2]=x[1]; x[1]=x[0]; x[0]=y*0.9+1e-3*(i&7);}
  double tp=(now()-t0)/N; N=2000000; t0=now();
  for(long i=0;i<N;i++){
    double z=0.3*(i&3)-0.4, Px[3], den=lam, g[3], e;
    for(int r=0;r<3;r++){Px[r]=P[r][0]*x[0]+P[r][1]*x[1]+P[r][2]*x[2]; den+=x[r]*Px[r];}
    for(int r=0;r<3;r++) g[r]=Px[r]/den;
    e=z-(A[0]*x[0]+A[1]*x[1]+A[2]*x[2]);
    for(int r=0;r<3;r++){A[r]+=g[r]*e; for(int c=0;c<3;c++) P[r][c]=(P[r][c]-g[r]*Px[c])/lam;}
    x[2]=x[1]; x[1]=x[0]; x[0]=z; sink+=A[0];
  }
  double tu=(now()-t0)/N;
  printf("%.3f %.3f\n", tp*1e9, tu*1e9); return (int)(sink*0);
}
