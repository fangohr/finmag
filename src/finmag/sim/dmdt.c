int dmdt(double alpha, double gamma, double c,
    int Mn, double* M, int Hn, double* H, int dMdtn, double* dMdt,
    int Pn, double* P);

#define DEBUG 1
#ifdef DEBUG
    #define DLOG(...) printf(__VA_ARGS__)
#else
    #define DLOG(...) /* nothing */
#endif

int dmdt(double alpha, double gamma, double c,
         int Mn, double* M, int Hn, double* H, int dMdtn, double* dMdt,
         int Pn, double* P) {
  
    if ( Mn != Hn || Mn != dMdtn ) {
        DLOG("in '%s': ", __PRETTY_FUNCTION__);
        DLOG("Arrays don't have the same number of entries. ");
        DLOG("M[%d], H[%d], dMdt[%d].\n", Mn, Hn, dMdtn);

        return EXIT_FAILURE;
    }

    const int DIMENSIONS = 3;

    if ( Mn % DIMENSIONS != 0 ) {
        DLOG("in '%s': ", __PRETTY_FUNCTION__);
        DLOG("Can't split arrays into %d dimensions. ", DIMENSIONS);
        DLOG("M[%d].\n", Mn);

        return EXIT_FAILURE;
    }

    const int ENTRIES_PER_DIM = Mn / DIMENSIONS;

    /* The first ENTRIES_PER_DIM entries correspond to the x-dimension,
    the second ENTRIES_PER_DIM entries to the y-dimension and the last
    bunch to the z-dimension. There could have been two variables called
    y_offset and z_offset, but X, Y and Z make for nicer array indexing. */

    const int X = 0;
    const int Y = ENTRIES_PER_DIM;
    const int Z = 2 * ENTRIES_PER_DIM;

    double p = gamma / (1 + alpha*alpha); /* precession factor of the LLG */
    double q = gamma * alpha / (1 + alpha*alpha); /* damping */

    for ( int i=0; i<ENTRIES_PER_DIM; i++ ) {
        double MM = M[X+i]*M[X+i] + M[Y+i]*M[Y+i] + M[Z+i]*M[Z+i];

        dMdt[X+i] =
            - p * (M[Y+i]*H[Z+i] - M[Z+i]*H[Y+i])
            - q * (  M[Y+i] * (M[X+i]*H[Y+i] - M[Y+i]*H[X+i])
                   - M[Z+i] * (M[Z+i]*H[X+i] - M[X+i]*H[Z+i]))
            - c * (MM - 1) * M[X+i];
        dMdt[Y+i] =
            - p * (M[Z+i]*H[X+i] - M[X+i]*H[Z+i])
            - q * (  M[Z+i] * (M[Y+i]*H[Z+i] - M[Z+i]*H[Y+i])
                   - M[X+i] * (M[X+i]*H[Y+i] - M[Y+i]*H[X+i]))
            - c * (MM - 1) * M[Y+i];
        dMdt[Z+i] =
            - p * (M[X+i]*H[Y+i] - M[Y+i]*H[X+i])
            - q * (  M[X+i] * (M[Z+i]*H[X+i] - M[X+i]*H[Z+i])
                   - M[Y+i] * (M[Y+i]*H[Z+i] - M[Z+i]*H[Y+i]))
            - c * (MM - 1) * M[Z+i];
    }

    for ( int i=0; i<Pn; i++ ) {
        /* pin the magnetisation at the given points by setting dM/dt to 0. */
        int node = int(P[i]);
        dMdt[X+node] = 0;
        dMdt[Y+node] = 0;
        dMdt[Z+node] = 0;
    }

    return EXIT_SUCCESS;
}
