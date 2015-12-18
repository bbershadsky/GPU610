# GPU610
GenPrime 3.5
Prime number generator using CUDA

This is the final version of a CUDA optimized prime number generator
It accepts command line range input (3~500000) and/or an interactive 3 step process to generate numbers very quickly.
A log file is saved to the executable's directory with a timestamp every time the program is run.

A regular CPU-based generator is also included for performance comparison.

On a i7-4790K @ 4.00Ghz and Nvidia GTX970 desktop the program can generate 500,000 prime numbers in under 60 milliseconds using
the Sieve of Eratosthenes, over a 4000% improvement in speed compared to a CPU-only argorithm.
