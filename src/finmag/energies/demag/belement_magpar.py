import instant

def return_bele_magpar():
    args = [["n_bvert", "bvert", "in"],["facv1_n", "facv1", "in"],["facv2_n", "facv2", "in"], ["facv3_n", "facv3", "in"],["matele_n","matele"]]
    return instant.inline_with_numpy(C_CODE, arrays=args)

C_CODE="""
int Bele(int n_bvert,double* bvert,int facv1_n, double* facv1, int facv2_n, double* facv2,
      int facv3_n,double* facv3,int matele_n,double* matele);

#define ND 3       /**< space dimensions (no of cartesian coordinates) */
#define NV 4       /**< number of vertices(=degrees of freedom) per element */
#define NF 4       /**< number of faces per element */
#define NN 3       /**< number of vertices per face */


#define C_BND -1   /**< indicator for boundary node/face */
#define C_INT -2   /**< indicator for interior node/face */
#define C_UNK -4   /**< indicator for unknown state */

#define D_EPS 1e-14 /**< threshold for equality of two real numbers */
#define PETSC_PI 3.1415926535897932384626433832795L


#define my_daxpy(a,b,c,d,e,f) {(e)[0]+=b*(c)[0];(e)[1]+=b*(c)[1];(e)[2]+=b*(c)[2];}
#define my_dcopy(a,b,c,d,e)   {(d)[0]=(b)[0];(d)[1]=(b)[1];(d)[2]=(b)[2];}
#define my_dnrm2(a,b,c)       sqrt((b)[0]*(b)[0]+(b)[1]*(b)[1]+(b)[2]*(b)[2])
#define my_dscal(a,b,c,d)     {(c)[0]*=b;(c)[1]*=b;(c)[2]*=b;}
#define my_ddot(a,b,c,d,e)    ((b)[0]*(d)[0]+(b)[1]*(d)[1]+(b)[2]*(d)[2])
#define douter(a,b,c,d)       {(d)[0]=(b)[1]*(c)[2]-(b)[2]*(c)[1];(d)[1]=(b)[2]*(c)[0]-(b)[0]*(c)[2];(d)[2]=(b)[0]*(c)[1]-(b)[1]*(c)[0];}

int PointFromPlane(double *x, double *v1, double *v2, double *v3, double *d)
{
  // computes the distance beetween the point x and the plane defined by v1, v2, v3
  // note that x, v1, v2 and v3 are 3-dimensional arrays (pointer to double)

  double  ab[ND],ac[ND]; // vectors ab and ac
  double  n[ND];         // vector n, normal to the plane

  /* calculate edge vectors */
  my_dcopy(ND,v1,1,ab,1);         // ab is now v1
  my_daxpy(ND,-1.0,v2,1,ab,1);    // ab = ab - v2
  my_dcopy(ND,v1,1,ac,1);         // ac is now v1
  my_daxpy(ND,-1.0,v3,1,ac,1);    // ac = ac - v3
  // summary: ab = v1 - v2
  //          ac = v1 - v3

  /* calculate normal vector */
  douter(ND,ab,ac,n);             // n = cross(ab, ac)

  /* calculate distance */
  // normally, this would have to be divided by norm(n), because n is not a unit vector
  *d=my_ddot(ND,x,1,n,1)-my_ddot(ND,v1,1,n,1); // d = x \dot n - v1 \dot n
                                               // or (x-v1) \dot n
  return(0);
}

int Bele(int n_bvert,double* bvert,int facv1_n, double* facv1, int facv2_n, double* facv2,
      int facv3_n,double* facv3,int matele_n,double* matele)
{
  double   *rr,zeta[ND],zetal;
  double   rho1[ND],rho2[ND],rho3[ND];
  double   rho1l,rho2l,rho3l;
  double   s1,s2,s3;
  double   eta1[ND],eta2[ND],eta3[ND];
  double   eta1l,eta2l,eta3l;
  double   xi1[ND],xi2[ND],xi3[ND];
  double   gamma1[ND],gamma2[ND],gamma3[ND];
  double   p[ND],a,omega;

  matele[0]=matele[1]=matele[2]=0.0;

  /* get coordinates of face's vertices */
  my_dcopy(ND,facv1,1,rho1,1);
  my_dcopy(ND,facv2,1,rho2,1);
  my_dcopy(ND,facv3,1,rho3,1);

  /* calculate edge vectors and store them in xi_j */
  my_dcopy(ND,rho2,1,xi1,1);
  my_daxpy(ND,-1.0,rho1,1,xi1,1);
  my_dcopy(ND,rho3,1,xi2,1);
  my_daxpy(ND,-1.0,rho2,1,xi2,1);
  my_dcopy(ND,rho1,1,xi3,1);
  my_daxpy(ND,-1.0,rho3,1,xi3,1);

  /* calculate zeta direction */
  douter(ND,xi1,xi2,zeta);

  /* calculate area of the triangle */
  zetal=my_dnrm2(ND,zeta,1);
  a=0.5*zetal;

  /* renorm zeta */
  my_dscal(ND,1.0/zetal,zeta,1);

  /* calculate s_j and normalize xi_j */
  s1=my_dnrm2(ND,xi1,1);
  my_dscal(ND,1.0/s1,xi1,1);
  s2=my_dnrm2(ND,xi2,1);
  my_dscal(ND,1.0/s2,xi2,1);
  s3=my_dnrm2(ND,xi3,1);
  my_dscal(ND,1.0/s3,xi3,1);

  douter(ND,zeta,xi1,eta1);
  douter(ND,zeta,xi2,eta2);
  douter(ND,zeta,xi3,eta3);

  gamma1[0]=gamma3[1]=my_ddot(ND,xi2,1,xi1,1);
  gamma1[1]=my_ddot(ND,xi2,1,xi2,1);
  gamma1[2]=gamma2[1]=my_ddot(ND,xi2,1,xi3,1);

  gamma2[0]=gamma3[2]=my_ddot(ND,xi3,1,xi1,1);
  gamma2[2]=my_ddot(ND,xi3,1,xi3,1);

  gamma3[0]=my_ddot(ND,xi1,1,xi1,1);

  /* get R=rr */
  rr=bvert;

  double d;
  PointFromPlane(rr,rho1,rho2,rho3,&d);
  if (fabs(d)<D_EPS) return(0);

  /* calculate rho_j */
  my_daxpy(ND,-1.0,rr,1,rho1,1);
  my_daxpy(ND,-1.0,rr,1,rho2,1);
  my_daxpy(ND,-1.0,rr,1,rho3,1);

  /* zetal gives ("normal") distance of R from the plane of the triangle */
  zetal=my_ddot(ND,zeta,1,rho1,1);

  /* skip the rest if zetal==0 (R in plane of the triangle)
     -> omega==0 and zetal==0 -> matrix entry=0
  */
  if (fabs(zetal)<=D_EPS) {
    return(0);
  }

  rho1l=my_dnrm2(ND,rho1,1);
  rho2l=my_dnrm2(ND,rho2,1);
  rho3l=my_dnrm2(ND,rho3,1);

  double t_nom,t_denom;
  t_nom=
    rho1l*rho2l*rho3l+
    rho1l*my_ddot(ND,rho2,1,rho3,1)+
    rho2l*my_ddot(ND,rho3,1,rho1,1)+
    rho3l*my_ddot(ND,rho1,1,rho2,1);
  t_denom=
    sqrt(2.0*
      (rho2l*rho3l+my_ddot(ND,rho2,1,rho3,1))*
      (rho3l*rho1l+my_ddot(ND,rho3,1,rho1,1))*
      (rho1l*rho2l+my_ddot(ND,rho1,1,rho2,1))
    );

  /* catch special cases where the argument of acos
     is close to -1.0 or 1.0 or even outside this interval

     use 0.0 instead of D_EPS?
     fixes problems with demag field calculation
     suggested by Hiroki Kobayashi, Fujitsu
  */
  if (t_nom/t_denom<-1.0) {
    omega=(zetal >= 0.0 ? 1.0 : -1.0)*2.0*M_PI;
  }
  /* this case should not occur, but does - e.g. examples1/headfield */
  else if (t_nom/t_denom>1.0) {
    return(0);
  }
  else {
    omega=(zetal >= 0.0 ? 1.0 : -1.0)*2.0*acos(t_nom/t_denom);
  }

  eta1l=my_ddot(ND,eta1,1,rho1,1);
  eta2l=my_ddot(ND,eta2,1,rho2,1);
  eta3l=my_ddot(ND,eta3,1,rho3,1);

  p[0]=log((rho1l+rho2l+s1)/(rho1l+rho2l-s1));
  p[1]=log((rho2l+rho3l+s2)/(rho2l+rho3l-s2));
  p[2]=log((rho3l+rho1l+s3)/(rho3l+rho1l-s3));

  matele[0]=(eta2l*omega-zetal*my_ddot(ND,gamma1,1,p,1))*s2/(8.0*PETSC_PI*a);
  matele[1]=(eta3l*omega-zetal*my_ddot(ND,gamma2,1,p,1))*s3/(8.0*PETSC_PI*a);
  matele[2]=(eta1l*omega-zetal*my_ddot(ND,gamma3,1,p,1))*s1/(8.0*PETSC_PI*a);

  return(0);
}
"""
