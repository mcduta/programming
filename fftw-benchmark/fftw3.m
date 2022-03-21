
function fftw3(filename)

%
% defaults
%
M = 1;  N = 1;  L = 1;


%
% read in result file
%
fid = fopen(filename, 'rt');

% number of dimensions
d = textscan(fid,'%s',1+2,'delimiter',' ');
ndim = str2num(cell2mat(d{1}(end)));

% dimensions
d = textscan(fid,'%s',ndim+2,'delimiter',' ');
for idim=1:ndim
  vdim(idim) = str2num(cell2mat(d{1}(idim+2)));
end
if (ndim>0) M = vdim(1); end
if (ndim>1) N = vdim(2); end
if (ndim>2) L = vdim(3); end

% data
d = fscanf(fid, '%f', [4,M*N*L]);

fclose(fid);


%
% reshape data
%
u = complex(zeros(M,N,L), zeros(M,N,L));
U = u;
k = 0;

for m = 1:M
  for n = 1:N
   for l = 1:L
     k = k + 1;
     u(m,n,l) = complex(d(1,k),d(2,k));
     U(m,n,l) = complex(d(3,k),d(4,k));
    end
  end
end

u = squeeze(u);
U = squeeze(U);



%
% Matlab fft
%
U0 = fftn(u);


%
% compare results
%
nrm = norm(U(:)-U0(:))/sqrt(M*N*L);


%
% report
%
disp(sprintf(' %d dimensional DFT',ndim));
disp(sprintf(' size = %s',num2str(vdim)));
disp(sprintf(' rms diff: %e', nrm));


% junk %

% d = fscanf(fid, '%f', [4,M*N*L]);
% u = complex(d(1,:), d(2,:)); u = (squeeze(reshape(u',L,N,M)))';
% U = complex(d(3,:), d(4,:)); U = (squeeze(reshape(U',L,N,M)))';



%    note:    Matlab fft2 computes
%             U = fft(fft(u, [],1), [],2);

% end %
