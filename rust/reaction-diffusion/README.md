# Grid solution of the Gray-Scott model

## Overview
The Gray-Scott equations are a classic reaction-diffusion system of partial differential equations that simulates how two chemical substances react and spread across a space domain in time. The equations describe a reaction in which the substance $u$ and consumed and $v$ is produced
```math
\begin{aligned}
  \frac{\partial u}{\partial t} &= D_u \nabla^2 u - uv^2 + \alpha(1-u) \\
  \frac{\partial v}{\partial t} &= D_v \nabla^2 v + uv^2 - (\alpha+\beta)v
\end{aligned}
```
In the above
* $u$ and $v$ are the concentrations of the two substances;
* $D_u$ and $D_v$ are the diffusion rates of the substances;
* $\alpha$ is the "feed" rate at which substance $u$ is replenished externally and
* $\beta$ is the "kill/death" rate at which substance $v$ is removed from the system. 

As $u$ is being consumed, it needs replenishing, otherwise the reaction would stop and equally, as $v$ is being produced, it needs to be removed in order to maintain the reaction. Using only the 4 parameters to describe this chemical dynamics, the model generates complex, organic patterns like spots, stripes, and dividing cells.

## Numerical solution
The equations are discretised in space on a regular grid on a square domain using finite differences and the boundary conditions are periodic (up-down, left-right). A forward Euler time-integration is enough for interesting patterns to evolve.

The initial solution is 1 for $u$ and 0 for $v$ everywhere, except at a few spots (chosen randoml), at which $u$ is 0.5 and $v$ 0.25.

## Build and run
The project is built with
```
cargo build --release
```
and run with one of the parameter input files, for instance
```
./target/release/rd --nx 600 --ny 500 --nt 100000 params/coral.json
```

## Further reading
* P. Gray and S. K. Scott, *Chemical Oscillations and Instabilities: Non-linear Chemical Kinetics*, v. 21 of International Series of Monographs on Chemistry, 1994.
* https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/
* https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system


