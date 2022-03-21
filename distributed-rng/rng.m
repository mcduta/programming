

%
% ----- random number generator (MCG31m1)
%
clear all;

n = 1024;

a = 1132489760;
m = 2147483647; % 2^31-1

% seed: integral number in range 1...258
z = 1.0;

x(1) = z / m;
for k = 2:n
  z    = mod(a*z, m);
  x(k) = z / m;
end


k = floor(n/2); plot(x([1:k]),x(k+[1:k]),'k.'); axis equal;


%
% ----- random number generator (MRG32k3a)
%
% This generator passes all tests of G. Marsaglia's Diehard testsuite.
% Its period is (m1^3 - 1)(m2^3 - 1)/2 which is nearly 2^191.  The
% reference
%        P. L'Ecuyer, R. Simard, E. J. Chen, W. D. Kelton:
%        An Object-Oriented Random-Number Package With Many Long 
%        Streams and Substreams. 2001. Operations Research.
% reports: "This generator is well-behaved in all dimensions
% up to at least 45: ..." [with respect to the spectral test].

clear all;

n    = 1024;

% initialise sequence (MCG31m1)
a = 1132489760;
m = 2^31 - 1;
z = [1.0; 0.0];

z(:,2) = mod(a*z(:,1),m);
z(:,3) = mod(a*z(:,2),m);
z      = z / m;

% coefficients (MRG32k3a)
A = [
       0  1403580  -810728
  527612        0 -1370589
    ];
m = 2^32 - [209; 22853];

% pseudo-random sequence
for k = 4:n
  z(1,k) = mod(A(1,:)*[z(1,k-1); z(1,k-2); z(1,k-3)], m(1));
  z(2,k) = mod(A(2,:)*[z(2,k-1); z(2,k-2); z(2,k-3)], m(2));
  x(k)   = mod(z(1,k) - z(2,k), m(1));
  x(k)   = x(k) / m(1);
end

k = floor(n/2); plot(x([1:k]),x(k+[1:k]),'k.'); axis equal;


% end

