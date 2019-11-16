This folder contains *unpolished* codes for computing &rho;<sup>-1</sup><sub>r</sub>(0.5).

You may have to look into the code to manually change the configuration. Here are some notes that might help you understand the code (or revise it for other purposes):

1. large_count.py: the python file that computes the sizes of likelihood ratio regions

1.a. args.r is the radius
1.b. for scalability reason, I write the code in the way that partitions computations to 10 different jobs that can be run in parallel. args.p should be something in {0, 1,..., 9} as one of the 10 jobs. Please change line 18 to your dimension, line 19 to your K, and line 40 - 45 accordingly. 

2. get_threshold.py: it computes the \rho_r^-1(0.5) as the threshold

2.a. args.r_start and args.r_end are simply the range of radii that you want to compute. 
2.b. args.a is the alpha value * 10 (assuming alpha in {0.1, 0.2, ... 1.0})
2.c. args.p: if you didn't change large_count.py, I think it should be set to 10 (the number of partitions).
2.d. Please check line 21 - line 34 to specify the parameters in your own dataset. 
2.e. line 95 checks that a sufficient condition that the computation of likelihood ratio region sizes are correct.


