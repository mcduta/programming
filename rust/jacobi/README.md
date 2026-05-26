
# Jacobi update

## Overview
The Jacobi update is a popular choice for illustrating data parallelism, as it is a simple iterative algorithm with a stencil-based update.

## Implementation
This is a Rust implementation of the Jacobi update which uses

## Build and run
The code takes in the 2D matrix dimensions as command line arguments and reports the size (in GBytes) of the matrix allocation as well as the update performance measured in Gflops.

The usual Rust practice applies, e.g. the build and run can be done in one step with
```
cargo run --release -- --nx 30000 --ny 20000
```