
%
%     name:     HEAT_2D.M
%     synopsis: verify correctness of C implementation
%

clear all;

% grid parameters
I = 101;
J = I
N = 100;
nu= 0.25;

% wavenumbers: sin(wnx*pi*x)*sin(wny*pi*y)
wnx = 1;
wny = 2;

% grid spacing
dx = 1/(I-1);
dy = dx;

% initialise
u = zeros(I,J);
x = zeros(I,J);
y = zeros(I,J);

% initial solution
for i = 1:I
  for j = 1:J
    x(i,j) = (i-1)*dx;
    y(i,j) = (j-1)*dy;
    u(i,j) = sin(wnx*pi*x(i,j))*sin(wny*pi*y(i,j));
  end
end

% compute
for n = 1:N
  uo = u;
  for i = 2:I-1
    for j = 2:J-1
      u(i,j) = uo(i,j) + nu*(uo(i+1,j)+uo(i-1,j)+uo(i,j+1)+uo(i,j-1)-4*uo(i,j));
    end
  end
end

% final simulated time
T = N*nu*dx*dx;

% analytic solution
ua = sin(wnx*pi*x).*sin(wny*pi*y)*exp(-(wnx^2+wny^2)*pi^2*T);

% plots
figure(1); clf;
  subplot(211); mesh(x,y,u);  title('computed');
  subplot(212); mesh(x,y,ua); title('analytic');
figure(2); clf;
  j = round(J/4);
  plot(x(:,j),u(:,j),'b', x(:,j),ua(:,j),'g');
  legend('computed', 'analytic');

%
% ----- import sequential run data and compare
%
d  = load('heat.out');
us = reshape(d(:,3),I,J);
figure(1); clf; mesh(u);
figure(2); clf; mesh(us);
figure(3); clf; mesh(u-us);
disp(norm(u-us)/I);


%
% ----- import parallel run data and compare
%
d  = load('heat_mpi.out');
up = reshape(d(:,3),I,J);
figure(1); clf; mesh(u);
figure(2); clf; mesh(up);
figure(3); clf; mesh(u-up);
disp(norm(u-up)/I);



%
% ----- import parallel run data (XY partition) and compare
%
d = load('heat_mpi.out');
N = size(d,1);
if (N == I*J)
  up = zeros(size(u));
  for i = 1:I
    for j = 1:J
      coord = [x(i,j), y(i,j)];
      found = 0;
      k     = 0;
      while (~found & k<N)
        k = k + 1;
        if (norm(d(k,1:2)-coord) < 1.e-15)
          up(i,j) = d(k,3);
          found   = 1;
        end
      end
    end
  end
else
  disp(' *** error: size mismatch');
end

figure(1); mesh(u);
figure(2); mesh(up);
figure(3); mesh(u-up);
disp(norm(u-up)/I);



%======================================================================
%     alternate solutions:
%       1) explicit
%       2) implicit
%       3) Crank-Nocolson
%======================================================================

clear all;

% grid parameters
I = 101;
J = I;
N = 100;
nu= 0.12;

% wave numbers
wnx = 1;
wny = 2;

% grid spacing
dx = 1/(I-1);
dy = dx;

% initialise
u = zeros(I,J);
x = zeros(I,J);
y = zeros(I,J);

% initial solution
for i = 1:I
  for j = 1:J
    x(i,j) = (i-1)*dx;
    y(i,j) = (j-1)*dy;
    u(i,j) = sin(wnx*pi*x(i,j))*sin(wny*pi*y(i,j));
  end
end
u0 = u;

% vector/matrix size
M = (I-2)*(J-2);

% solution matrix
w = ones(M,1);
A = spdiags([w w -4*w w w], [-I+2 -1 0 +1 +I-2], M,M);

for j = 1:J-3 % account for boundaries
  k = j*(I-2);
  A(k,k+1) = 0;
  A(k+1,k) = 0;
end

clear w;

% final simulated time
T = N*nu*dx*dx;

% analytic solution
ua = sin(wnx*pi*x).*sin(wny*pi*y)*exp(-(wnx^2+wny^2)*pi^2*T);


%
% ----- explicit solution
%
v = reshape(u0(2:I-1, 2:J-1), M,1);

A2 = speye(M,M) + nu*A;
for n = 1:N
  v = A2*v;
end

% solution vector
u(2:I-1, 2:J-1) = reshape(v, I-2,J-2);

% plots
figure(1); clf;
  subplot(211); mesh(x,y,u);  title('computed');
  subplot(212); mesh(x,y,ua); title('analytic');
figure(2); clf;
  j = round(J/2);
  plot(x(:,j),u(:,j),'b', x(:,j),ua(:,j),'g');
  legend('computed', 'analytic');
  xlabel(sprintf('rms error = %10.6f', norm(u-ua)/I));
pause


%
% ----- implicit solution
%
v = reshape(u0(2:I-1, 2:J-1), M,1);

A2 = speye(M,M) - nu*A;
for n = 1:N
  v = A2\v;
end

% solution vector
u(2:I-1, 2:J-1) = reshape(v, I-2,J-2);

% plots
figure(1); clf;
  subplot(211); mesh(x,y,u);  title('computed');
  subplot(212); mesh(x,y,ua); title('analytic');
figure(2); clf;
  j = round(J/2);
  plot(x(:,j),u(:,j),'b', x(:,j),ua(:,j),'g');
  legend('computed', 'analytic');
  xlabel(sprintf('rms error = %10.6f', norm(u-ua)/I));

pause


%
% ----- Crank-Nicolson
%
v = reshape(u0(2:I-1, 2:J-1), M,1);

A2 = (speye(M,M) - 0.5*nu*A)\(speye(M,M) + 0.5*nu*A);
for n = 1:N
  v = A2*v;
end

% solution vector
u(2:I-1, 2:J-1) = reshape(v, I-2,J-2);

% plots
figure(1); clf;
  subplot(211); mesh(x,y,u);  title('computed');
  subplot(212); mesh(x,y,ua); title('analytic');
figure(2); clf;
  j = round(J/2);
  plot(x(:,j),u(:,j),'b', x(:,j),ua(:,j),'g');
  legend('computed', 'analytic');
  xlabel(sprintf('rms error = %10.6f', norm(u-ua)/I));


% end
