
%
%    name:     heat.m
%    synopsis: Matlab auxiliary code to test the
%              mpi/openmp/sequential heat solvers
%

%
% ----- explicit finite difference solution (FTCS)
%
clear all;

% parameters
J = 50;
N = 200;

% parameters
nu = 0.5;
dx = 1/(J-1);
dt = nu*dx^2;
T  = N*dt;

% initial conditions
x  = linspace(0,1,J);
u  = sin(pi*x);

% time-marching
for n = 1:N
  uo = u;

  u(1) = 0;
  for j = 2:J-1
    xj = (j-1)*dx;
    fj = sin(pi*xj);
    u(j) = uo(j) + nu*(uo(j-1)-2.0*uo(j)+uo(j+1));
  end
  u(end) = 0;
end

% plot results
plot(x,sin(pi*x)*exp(-pi^2*T),'g--', x,u,'b.');

% end
