function [g, gdata, gprior] = gpla_g(w, gp, x, y, varargin)
%GPLA_G   Evaluate gradient of Laplace approximation's marginal 
%         log posterior estimate (GPLA_E)
%
%  Description
%    G = GPLA_G(W, GP, X, Y, OPTIONS) takes a full GP parameter
%    vector W, structure GP a matrix X of input vectors and a
%    matrix Y of target vectors, and evaluates the gradient G of
%    EP's marginal log posterior estimate. Each row of X
%    corresponds to one input vector and each row of Y corresponds
%    to one target vector.
%
%    [G, GDATA, GPRIOR] = GPLA_G(W, GP, X, Y, OPTIONS) also returns
%    the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%  
%  See also
%    GP_SET, GP_G, GPLA_E, GPLA_PRED
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_G';
  ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
  ip.addRequired('gp',@isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(w, gp, x, y, varargin{:});
  z=ip.Results.z;

  gp = gp_unpak(gp, w);       % unpak the parameters
  [tmp,tmp,hier]=gp_pak(gp);   % Get the hierarchy of the parameters
  ncf = length(gp.cf);
  n=size(x,1);

  g = [];
  gdata = [];
  gprior = [];
  
  if isfield(gp, 'savememory') && gp.savememory
    savememory=1;
  else
    savememory=0;
  end
  
  % First Evaluate the data contribution to the error
  switch gp.type
      case 'FULL'
          % ============================================================
          % FULL
          % ============================================================
          
          if ~isfield(gp.lik, 'nondiagW')
              % Likelihoos with diagonal Hessian
              
          else
              % Likelihoods with non-diagonal Hessian
              
              switch gp.lik.type
                  case {'spatcluster'}
                      nout = gp.lik.RCPnum;
                  otherwise
                      [n,nout] = size(y);
              end
              if isfield(gp, 'comp_cf')  % own covariance for each ouput component
                  multicf = true;
                  if length(gp.comp_cf) ~= nout && nout > 1
                      error('GPLA_ND_G: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
                  end
              else
                  multicf = false;
              end
              
              switch gp.lik.type
                  
                  
                  case 'spatcluster'
                      
                      nout = gp.lik.RCPnum;
                      
                      % get the parameters of Laplace approximation
                      [e, ~, ~, p] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
                      %[f, LK, a, W] = deal(p.f, p.L, p.a, p.La2);
                      [f, LK, a, W, LB] = deal(p.f, p.L, p.a, p.La2, p.p);
                      if isnan(e)
                          g=NaN; gdata=NaN; gprior=NaN;
                          return;
                      end
                      
                      %  The implementation follows the general results of Rasmussen and
                      %  Williams (2006) (page 125, http://www.gaussianprocess.org/gpml/chapters/RW5.pdf).
                      %  The main difference is in how we calculate equation 5.23
                      
                      % help matrices and vectors
%                       V = full(LK'*W);  % !! this is full anyways and, hence, sparse matrix will be slower
%                       B = V*LK;
%                       B(1:n*nout+1:end)=B(1:n*nout+1:end)+1;
%                       [LB, notpositivedefinite] = chol(B, 'lower');
%                       if notpositivedefinite
%                           % instead of stopping to chol error, return NaN
%                           g=NaN;
%                           gdata = NaN;
%                           gprior = NaN;
%                           return;
%                       end
                      RW = LB\LK'; 
                      RW = RW'*RW;  % RW = inv(inv(K)+W);
                      R =  W - W*RW*W;
                      
                      DW = gp.lik.fh.llg3(gp.lik, y, f, 'latent', z) ;
                      s2 = 0.5*sum(reshape(DW(:,2).*RW(DW(:,1)),nout*nout,length(f)))';
                      %               s2 = zeros(size(f));
                      %               for h1=1:length(f)
                      %                   s2(h1) = 0.5*sum(sum( DW{h1}.*RW ));
                      %               end
                      
                      % =================================================================
                      % Gradient with respect to the covariance function parameters
                      if ~isempty(strfind(gp.infer_params, 'covariance'))
                          % Evaluate the gradients from covariance functions
                          i1=0;
                          for i=1:ncf
                              
                              % check in which components the covariance function
                              % is present
                              do = false(nout,1);
                              if multicf
                                  for z1=1:nout
                                      if any(gp.comp_cf{z1}==i)
                                          do(z1) = true;
                                      end
                                  end
                              else
                                  do = true(nout,1);
                              end
                              
                              gpcf = gp.cf{i};
                              DKffc = gpcf.fh.cfg(gpcf, x); % derivative of covariance matrix
                              np=length(DKffc);
                              gprior_cf = -gpcf.fh.lpg(gpcf);
                              
                              g1 = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
                              for i2 = 1:np
                                  %DKff=DKffc{i2};
                                  DKff = sparse(n*nout,n*nout);
                                  s1tmp=0;
                                  for z1=1:nout
                                      if do(z1)
                                          DKff((z1-1)*n+1:z1*n,(z1-1)*n+1:z1*n) = DKffc{i2};
                                          %s1tmp = s1tmp+sum(sum(R((z1-1)*n+1:z1*n,(z1-1)*n+1:z1*n).*DKffc{i2}));
                                      end
                                  end
                                  i1 = i1+1;
                                  s1 = 0.5 * a'*DKff*a - 0.5*sum(sum(R.*DKff));
                                  %s1 = 0.5 * a'*DKff*a - 0.5*s1tmp;
                                  b = DKff * g1;
                                  s3 = b - RW*(W*b); %s3 = (eye(size(K)) + K*W)\b;
                                  gdata(i1) = -(s1 + s2'*s3);
                              end
                              gprior=[gprior gprior_cf];
                          end
                      end
                      
                      % =================================================================
                      % Gradient with respect to likelihood function parameters
                      if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
                              && ~isempty(gp.lik.fh.pak(gp.lik))
                          %K = LK*LK';
                          
                          gdata_lik = 0;
                          lik = gp.lik;
                          
                          g_logPrior = -lik.fh.lpg(lik);
                          if ~isempty(g_logPrior)
                              
                              [DW_sigma, DW_sigmaIndex]= lik.fh.llg3(lik, y, f, 'latent2+param', z);
                              DL_sigma = lik.fh.llg(lik, y, f, 'param', z);
                              s3 = RW * lik.fh.llg2(lik, y, f, 'latent+param', z);
                              dBdth = squeeze(0.5*sum(DW_sigma.*RW(DW_sigmaIndex),1));
                              %dBdth = squeeze(0.5*sum(DW_sigma(:,2,:).*RW(DW_sigma(:,1,1)),1))';
                              
                              %dBdth = zeros(1,size(DW_sigma,3));
                              %for cc3=1:length(DW_sigma)
                              %dBdth(cc3) = 0.5*sum(DW_sigma{cc3}(:,2).*RW(DW_sigma{cc3}(:,1)));
                              %end
                              %                       for cc3=1:size(DW_sigma,3)
                              %                           dBdth(cc3) = 0.5*sum(DW_sigma(:,2,cc3).*RW(DW_sigma(:,1,cc3)));
                              %                       end
                              %                       for cc3=1:size(DW_sigma,3)
                              %                           dBdth(cc3) = 0.5.*sum(sum(RW.*DW_sigma(:,:,cc3)));
                              %                       end
                              
                              gdata_lik = - DL_sigma - dBdth - s2'*s3;
                              
                              % set the gradients into vectors that will be returned
                              gdata = [gdata gdata_lik];
                              gprior = [gprior g_logPrior];
                              i1 = length(g_logPrior);
                              i2 = length(gdata_lik);
                              if i1  > i2
                                  gdata = [gdata zeros(1,i1-i2)];
                              end
                              
                          end
                          
                      end
                      
                  otherwise
                      
                      
              end
              
              
          end
          
  end
  
  % If ther parameters of the model (covariance function parameters,
  % likelihood function parameters, inducing inputs) have additional
  % hyperparameters that are not fixed,
  % set the gradients in correct order
  if length(gprior) > length(gdata)
      %gdata(gdata==0)=[];
      tmp=gdata;
      gdata = zeros(size(gprior));
      % Set the gradients to right place
      if any(hier==0)
          gdata([hier(1:find(hier==0,1)-1)==1 ...  % Covariance function
              hier(find(hier==0,1):find(hier==0,1)+length(g_logPrior)-1)==0 ... % Likelihood function
              hier(find(hier==0,1)+length(g_logPrior):end)==1]) = tmp;  % Inducing inputs
      else
          gdata(hier==1)=tmp;
      end
  end
  g = gdata + gprior;
  
  assert(isreal(gdata))
  assert(isreal(gprior))
end
