import instant

def return_csa_magpar():
    args = [["xn", "x", "in"], ["v1n", "v1", "in"], ["v2n", "v2", "in"], ["v3n", "v3", "in"]] 
    return instant.inline_with_numpy(C_CODE, arrays=args)

C_CODE="""
double SolidAngle(int xn, double *x, int v1n, double *v1, int v2n, double *v2, int v3n, double *v3);

#define my_daxpy(a,b,c,d,e,f) {(e)[0]+=b*(c)[0];(e)[1]+=b*(c)[1];(e)[2]+=b*(c)[2];}
#define my_dcopy(a,b,c,d,e)   {(d)[0]=(b)[0];(d)[1]=(b)[1];(d)[2]=(b)[2];}
#define my_dnrm2(a,b,c)       sqrt((b)[0]*(b)[0]+(b)[1]*(b)[1]+(b)[2]*(b)[2])
#define my_dscal(a,b,c,d)     {(c)[0]*=b;(c)[1]*=b;(c)[2]*=b;}
#define my_ddot(a,b,c,d,e)    ((b)[0]*(d)[0]+(b)[1]*(d)[1]+(b)[2]*(d)[2])
#define douter(a,b,c,d)       {(d)[0]=(b)[1]*(c)[2]-(b)[2]*(c)[1];(d)[1]=(b)[2]*(c)[0]-(b)[0]*(c)[2];(d)[2]=(b)[0]*(c)[1]-(b)[1]*(c)[0];}

// was PETSC_MACHINE_EPSILON*100 which yields 1.e-12 using doubles.
#define D_EPS                 1.e-12

int ND=3;
const double PETSC_PI=atan2(0.0, -1.0);

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

double SolidAngle(int xn, double *x, int v1n, double *v1, int v2n, double *v2, int v3n, double *v3)
{
  double omega;

  /* http://en.wikipedia.org/wiki/Solid_angle */

  double d;
  PointFromPlane(x,v1,v2,v3,&d);
  if (abs(d)<D_EPS) {
    omega=0.0;
    return(0);
  }

  double  t_ea[ND],t_eb[ND],t_ec[ND];
  double  t_nab[ND],t_nbc[ND],t_nca[ND];
  double  t_norm;

  /* calculate edge vectors */
  my_dcopy(ND,v1,1,t_ea,1);
  my_daxpy(ND,-1.0,x,1,t_ea,1);
  my_dcopy(ND,v2,1,t_eb,1);
  my_daxpy(ND,-1.0,x,1,t_eb,1);
  my_dcopy(ND,v3,1,t_ec,1);
  my_daxpy(ND,-1.0,x,1,t_ec,1);

  /* calculate normal vectors */
  douter(ND,t_ea,t_eb,t_nab);
  douter(ND,t_eb,t_ec,t_nbc);
  douter(ND,t_ec,t_ea,t_nca);

  /* normalize vectors */

  t_norm=my_dnrm2(ND,t_nab,1);
  if (t_norm < D_EPS) {
    omega=0.0;
    return(omega);
  }
  my_dscal(ND,1.0/t_norm,t_nab,1);

  t_norm=my_dnrm2(ND,t_nbc,1);
  if (t_norm < D_EPS) {
    omega=0.0;
    return(omega);
  }
  my_dscal(ND,1.0/t_norm,t_nbc,1);

  t_norm=my_dnrm2(ND,t_nca,1);
  if (t_norm < D_EPS) {
    omega=0.0;
    return(omega);
  }
  my_dscal(ND,1.0/t_norm,t_nca,1);

  /* calculate dihedral angles between facets */
  /* TODO source of this formula ? */

  double t_a_abbc,t_a_bcca,t_a_caab;
  t_a_abbc=t_nab[0]*t_nbc[0]+t_nab[1]*t_nbc[1]+t_nab[2]*t_nbc[2];
  t_a_bcca=t_nbc[0]*t_nca[0]+t_nbc[1]*t_nca[1]+t_nbc[2]*t_nca[2];
  t_a_caab=t_nca[0]*t_nab[0]+t_nca[1]*t_nab[1]+t_nca[2]*t_nab[2];
  if (t_a_abbc>1) t_a_abbc=PETSC_PI; else if (t_a_abbc<-1) t_a_abbc=0; else t_a_abbc=PETSC_PI-acos(t_nab[0]*t_nbc[0]+t_nab[1]*t_nbc[1]+t_nab[2]*t_nbc[2]);
  if (t_a_bcca>1) t_a_bcca=PETSC_PI; else if (t_a_bcca<-1) t_a_bcca=0; else t_a_bcca=PETSC_PI-acos(t_nbc[0]*t_nca[0]+t_nbc[1]*t_nca[1]+t_nbc[2]*t_nca[2]);
  if (t_a_caab>1) t_a_caab=PETSC_PI; else if (t_a_caab<-1) t_a_caab=0; else t_a_caab=PETSC_PI-acos(t_nca[0]*t_nab[0]+t_nca[1]*t_nab[1]+t_nca[2]*t_nab[2]);

  omega=t_a_abbc+t_a_bcca+t_a_caab-PETSC_PI;

  return(omega);
}
"""
