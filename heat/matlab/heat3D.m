
%
%     name:     HEAT.M
%     synopsis: * 3D time-dependent heat (difussion) equation solved
%                 using explicit finite-differencing
%               * Matlab code is used to verify correctness of C
%                 implementation
%     notes:    * the PDE is du/dt = div(curl(u))
%               * the explicit FD scheme is
%                 u(i,j,k) = u(i,j,k) + nu*(u(i+1,j,k)-2*u(i,j,k)+u(i-1,j,k)
%                                         + u(i,j+1,k)-2*u(i,j,k)+u(i,j-1,k)
%                                         + u(i,j,k+1)-2*u(i,j,k)+u(i,j,k-1))
%               * the explicit scheme is 2nd order in space, first
%                 order in time
%

clear all;

% grid parameters
I  = 101;
J  = I;
K  = I;
N  = 400;
nu = 0.15;

% wavenumbers: sin(wnx*pi*x)*sin(wny*pi*y)*sin(wnz*pi*z)
wnx = 1;
wny = 4;
wnz = 2;

% grid spacing
dx = 1/(I-1);
dy = dx;
dz = dx;

% final simulated time
T = N*nu*dx*dx;

% analytic solution exponent
expT = exp(-(wnx^2+wny^2+wnz^2)*pi^2*T);

% initialise
u = zeros(I,J,K);
x = zeros(I,J,K);
y = zeros(I,J,K);
z = zeros(I,J,K);

% initial solution and final solution (without time factor)
for k = 1:K
  for j = 1:J
    for i = 1:I
      x(i,j,k) = (i-1)*dx;
      y(i,j,k) = (j-1)*dy;
      z(i,j,k) = (k-1)*dz;
      % initial solution
      u(i,j,k) = sin(wnx*pi*x(i,j,k))*sin(wny*pi*y(i,j,k))*sin(wnz*pi*z(i,j,k));
      % analytic final solution
      ua(i,j,k) = u(i,j,k) * expT;
    end
  end
end


% compute
for n = 1:N
  uo = u;
  for k = 2:K-1
    for j = 2:J-1
      for i = 2:I-1
        u(i,j,k) = uo(i,j,k) + nu * ( ...
                                      uo(i+1,j,k) - 2.0*uo(i,j,k) + uo(i-1,j,k) ...
                                    + uo(i,j+1,k) - 2.0*uo(i,j,k) + uo(i,j-1,k) ...
                                    + uo(i,j,k+1) - 2.0*uo(i,j,k) + uo(i,j,k-1) ...
                                    );
      end
    end
  end
end


% rms error
rms = sqrt(sum((u(:)-ua(:)).^2)*(dx*dy*dz));

% error at one point
i = round(I/4);
j = i;
k = i;
err = abs(u(i,j,k) - ua(i,j,k));

disp([rms err]);


% plots
figure(1); clf;
k = round(K/4);
subplot(211); mesh(x(:,:,k),y(:,:,k),u(:,:,k));  title('computed');
subplot(212); mesh(x(:,:,k),y(:,:,k),ua(:,:,k)); title('analytic');

% end
