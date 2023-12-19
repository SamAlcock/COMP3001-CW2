//-------------------- COMP3001 OPENMP COURSEWORK - Report Part1 -----------------------------
//compile with gcc Question1.c -o p -O3 -lm -Wall -fopenmp

/*
* Initial Routine 1 output = 3.141494
* Routine 1 = 3.498754 secs
* 
* Initial Routine 2 outputs = 3.753276e+05, 1.126173e+15, 3.096350e+26
* Routine 2 = 1.880216 secs
*/

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

#define TRIALS 100000000 //you can amend this value if you want routine1() to run slower/faster.
#define N 1024*8
#define MAX_THREADS 16

double drandom();
void seed(double low_in, double hi_in);
void routine1(long long int num_trials);
void routine2();
void execute();
void initialize();
double u_exact(double x, double y);

double A[N][N];

long long int ADDEND = 150889;
unsigned long long MULTIPLIER = 764261123;
unsigned long long PMOD = 2147483647;
unsigned long long mult_n = 0;
double random_low, random_hi;
unsigned long long pseed[MAX_THREADS][4]; //[4] to padd to cache line
unsigned long long random_last = 0;
#pragma omp threadprivate(random_last)

int main() {

    double wtime;


    wtime = omp_get_wtime();

    routine1(TRIALS);

    wtime = omp_get_wtime() - wtime;

    printf("\n   Routine 1 took %f secs \n", wtime);

    //-----------------------------------------------------------------------------------------

    wtime = omp_get_wtime();

    routine2();

    wtime = omp_get_wtime() - wtime;

    printf("\n   Routine 2 took %f secs \n", wtime);

    return 0;
}

void routine1(long long int num_trials) {

    long long int i;  long long int Ncirc = 0;
    double pi, x, y, test;
    double r = 1.0;   // radius of circle. Side of squrare is 2*r 



    //START THE PARALLEL REGION
    omp_set_num_threads(8);
#pragma omp parallel
    seed(-r, r);  // generate the seed

    //PARALLELIZE THIS LOOP
#pragma omp for private(i,x,y,test) shared(Ncirc) schedule(static) 
    for (i = 0;i < num_trials; i++)
    {

        x = drandom(); //generate a random value
        y = drandom(); //generate a random value

        test = x * x + y * y;

        if (test <= r * r)
            Ncirc++;
    }

    //END THE PARALLEL REGION

    pi = 4.0 * ((double)Ncirc / (double)num_trials);

    //THE PI VALUE OF THE PARALLEL VERSION MIGHT BE SLIGHTLY DIFFERENT (THE 4TH, 5TH, 6TH DIGITS AFTER THE RADIX POINT MIGHT BE DIFFERENT); THIS IS BECAUSE DIFFERENT RANDOM VALUES ARE GENERATED IN x,y VARIABLES. 
    printf("\n Routine 1 : after %lld trials, pi is %lf \n", num_trials, pi);

}

//DO NOT AMEND THIS ROUTINE
double drandom()
{
    unsigned long long random_next;
    double ret_val;

    // 
    // compute an integer random number from zero to mod
    //
    random_next = (unsigned long long)((mult_n * random_last) % PMOD);
    random_last = random_next;

    //
    // shift into preset range
    //
    ret_val = ((double)random_next / (double)PMOD) * (random_hi - random_low) + random_low;
    return ret_val;
}

//DO NOT AMEND THIS ROUTINE
// set the seed and the range
void seed(double low_in, double hi_in)
{
    int i, id, nthreads;
    unsigned long long iseed;
    id = omp_get_thread_num();

#pragma omp single
    {
        if (low_in < hi_in)
        {
            random_low = low_in;
            random_hi = hi_in;
        }
        else
        {
            random_low = hi_in;
            random_hi = low_in;
        }

        //
        // The Leapfrog method ... adjust the multiplier so you stride through
        // the sequence by increments of "nthreads" and adust seeds so each 
        // thread starts with the right offset
        //

        nthreads = omp_get_num_threads();
        iseed = PMOD / MULTIPLIER;     // just pick a reasonable seed
        pseed[0][0] = iseed;
        mult_n = MULTIPLIER;
        for (i = 1; i < nthreads; ++i)
        {
            iseed = (unsigned long long)((MULTIPLIER * iseed) % PMOD);
            pseed[i][0] = iseed;
            mult_n = (mult_n * MULTIPLIER) % PMOD;
        }

    }
    random_last = (unsigned long long) pseed[id][0];
}


//PARALLELIZE AND VECTORIZE THIS ROUTINE USING OPENMP
void initialize() {

    unsigned int i, j;
    double x;
    omp_set_num_threads(8);
#pragma omp parallel for private(x,i,j) schedule(static)
    for (i = 0;i < N;i++) {
#pragma omp simd aligned(A:64)
        for (j = 0;j < N;j++) {
            x = (double)(i % 99) * (j % 87) + 0.043;
            A[i][j] = sqrt(x);
        }
    }

}

//PARALLELIZE AND VECTORIZE THIS ROUTINE USING OPENMP
void execute() {

    double  x, y, u_true;
    double u_true_norm = 0.0, error_norm = 0.0, u_norm = 0.0;

    unsigned int i, j;
    omp_set_num_threads(8);
#pragma omp parallel for private(u_norm,i,j) shared(A) schedule(static)
    for (i = 0;i < N;i++) {
#pragma omp simd aligned(A:64)
        for (j = 0;j < N;j++) {
            u_norm = u_norm + A[i][j] * A[i][j];
        }
    }

    u_norm = sqrt(u_norm);

#pragma omp parallel for private(x,y,u_true,error_norm,u_true_norm,i,j) shared(A) schedule(static)
    for (i = 0;i < N;i++) {
#pragma omp simd aligned(A:64)
        for (j = 0;j < N;j++) {
            x = (double)(2 * i - N + 1) / (double)(N - 1);
            y = (double)(2 * j - N + 1) / (double)(N - 1);
            u_true = u_exact(x, y);
            error_norm = error_norm + sqrt((A[i][j] - u_true) * (A[i][j] - u_true));
            u_true_norm = u_true_norm + u_true * u_true;
        }
    }

    error_norm = sqrt(error_norm);
    u_true_norm = sqrt(u_true_norm);

    printf("\n Routine2 : output is %e, %e and %e\n", u_norm, error_norm, u_true_norm);

}


double u_exact(double x, double y) {
    double value;

    value = (1.0 - x * x) * (1.0 - y * y);

    return value;
}

void routine2() {

    initialize();
    execute();

}


