function [Eft, Varft, lpyt, Eyt, Varyt] = gpla_pred(gp, x, y, varargin)
%GPLA_PRED  Predictions with Gaussian Process Laplace approximation
%
%  Description
%    [EFT, VARFT] = GPLA_PRED(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution at test inputs XT. Returns a posterior
%    mean EFT and variance VARFT of latent variables and the
%    posterior predictive mean EYT and variance VARYT.
%
%    [EFT, VARFT, LPYT] = GPLA_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be a vector.
% 
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPLA_PRED(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPLA_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPYT of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = [])
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%      fcorr  - Method used for latent marginal posterior corrections. 
%               Default is 'off'. For Laplace possible methods are
%               'fact' and 'cm2'.  If method is 'on', 'cm2' is used
%               for Laplace.
%
%    NOTE! In case of FIC and PIC sparse approximation the
%    prediction for only some PREDCF covariance functions is just
%    an approximation since the covariance functions are coupled in
%    the approximation and are not strictly speaking additive
%    anymore.
%
%    For example, if you use covariance such as K = K1 + K2 your
%    predictions Eft1 = ep_pred(GP, X, Y, X, 'predcf', 1) and 
%    Eft2 = ep_pred(gp, x, y, x, 'predcf', 2) should sum up to 
%    Eft = ep_pred(gp, x, y, x). That is Eft = Eft1 + Eft2. With 
%    FULL model this is true but with FIC and PIC this is true only 
%    approximately. That is Eft \approx Eft1 + Eft2.
%
%    With CS+FIC the predictions are exact if the PREDCF covariance
%    functions are all in the FIC part or if they are CS
%    covariances.
%
%    NOTE! When making predictions with a subset of covariance
%    functions with FIC approximation the predictive variance can
%    in some cases be ill-behaved i.e. negative or unrealistically
%    small. This may happen because of the approximative nature of
%    the prediction.
%
%  See also
%    GPLA_E, GPLA_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2012 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_PRED';
  ip.addRequired('gp', @isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
  ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('predcf', [], @(x) isempty(x) || iscell(x) ||...
                   isvector(x) && isreal(x) && all(isfinite(x)&x>0))
  ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                   (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
  ip.addParamValue('fcorr', 'off', @(x) ismember(x, {'off', ...
                   'cm2', 'fact', 'on','lr'}))
  if numel(varargin)==0 || isnumeric(varargin{1})
    % inputParser should handle this, but it doesn't
    ip.parse(gp, x, y, varargin{:});
  else
    ip.parse(gp, x, y, [], varargin{:});
  end
  xt=ip.Results.xt;
  yt=ip.Results.yt;
  z=ip.Results.z;
  zt=ip.Results.zt;
  predcf=ip.Results.predcf;
  tstind=ip.Results.tstind;
  fcorr=ip.Results.fcorr;
  if isempty(xt)
    xt=x;
    if isempty(tstind)
      if iscell(gp)
        gptype=gp{1}.type;
      else
        gptype=gp.type;
      end
      switch gptype
        case {'FULL' 'VAR' 'DTC' 'SOR'}
          tstind = [];
        case {'FIC' 'CS+FIC'}
          tstind = 1:size(x,1);
        case 'PIC'
          if iscell(gp)
            tstind = gp{1}.tr_index;
          else
            tstind = gp.tr_index;
          end
      end
    end
    if isempty(yt)
      yt=y;
    end
    if isempty(zt)
      zt=z;
    end
  end

  [tn, tnin] = size(x);
  
  switch gp.type
    case 'FULL'
      % ============================================================
      % FULL
      % ============================================================
      if ~isfield(gp.lik, 'nondiagW')

      else
        % Likelihoods with non-diagonal Hessian
                        
        switch gp.lik.type
                    
            case 'spatcluster'
                nout = gp.lik.RCPnum;
                n = size(y,1);
                
                if isfield(gp, 'comp_cf')  % own covariance for each ouput component
                    multicf = true;
                    if length(gp.comp_cf) ~= nout && nout > 1
                        error('GPLA_ND_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
                    end
                    if ~isempty(predcf)
                        if ~iscell(predcf) || length(predcf)~=nout && nout > 1
                            error(['GPLA_ND_PRED: if own covariance for each output component is used,'...
                                'predcf has to be cell array and contain nout (vector) elements.   '])
                        end
                    else
                        predcf = gp.comp_cf;
                    end
                else
                    multicf = false;
                    for i1=1:nout
                        predcf2{i1} = predcf;
                    end
                    predcf=predcf2;
                end
                                
                % get the parameters of Laplace approximation
                [~, ~, ~, p] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
                [f, LK, W, LB] = deal(p.f, p.L, p.La2, p.p);
                deriv = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
  
                V = LK'*W;
                iRV = (LB\V);
                
                % Calculate the covariance between training and prediction set
                % notice the order xt,x to avoid transpose later
                %  Do prediction in pieces in order to save memory
                xt_orig = xt;
                ntest_orig = size(xt_orig,1);
                nattime = floor(5000/nout);
                Eft = zeros(ntest_orig,nout);
                Covf = zeros(nout,nout,ntest_orig);
                for ss1 = 1:ceil(ntest_orig/nattime)
                    indss1 = 1+(ss1-1)*nattime:min(ntest_orig,ss1*nattime);
                    xt = xt_orig(indss1,:);
                    ntest = size(xt,1);
                    K_nf = zeros(ntest*nout,n*nout);
                    for n1=1:nout
                        if ~isempty(predcf{n1})
                            K_nf(1+(n1-1)*ntest:n1*ntest,1+(n1-1)*n:n1*n) = gp_cov(gp,xt,x,predcf{n1});
                        end
                    end
                    % Calculate the mean
                    Eftt = K_nf*deriv;
                    Eftt = reshape(Eftt,ntest,nout);
                    Eft(indss1,:) = Eftt;
                                        
                    R2 = iRV*K_nf';
                    for i1=1:nout
                        indi1 = (i1-1)*ntest+1:i1*ntest;
                        for i2=1:nout
                            indi2 = (i2-1)*ntest+1:i2*ntest;
                            Covft = - sum(K_nf(indi1,:)'.*(W*K_nf(indi2,:)'),1)' + sum(R2(:,indi1).*R2(:,indi2),1)';
                            if i1==i2
                                if ~isempty(predcf{i1})
                                    Covft = Covft + gp_trvar(gp,xt,predcf{i1});
                                end
                            end
                            Covf(i1,i2,indss1) = Covft;
                        end
                    end
                end
                
          otherwise
            
            
            
        end
        if nargout > 1
          Varft=Covf;
        end
        
      end
      
      
  end  
  if ~isequal(fcorr, 'off')
    % Do marginal corrections for samples
    [pc_predm, fvecm] = gp_predcm(gp, x, y, xt, 'z', z, 'ind', 1:size(xt,1), 'fcorr', fcorr);
    for i=1:size(xt,1)
      % Remove NaNs and zeros
      pc_pred=pc_predm(:,i);
      dii=isnan(pc_pred)|pc_pred==0;
      pc_pred(dii)=[];
      fvec=fvecm(:,i);
      fvec(dii)=[];
      % Compute mean correction
      Eft(i) = trapz(fvec.*(pc_pred./sum(pc_pred)));
    end
   end
  % ============================================================
  % Evaluate also the predictive mean and variance of new observation(s)
  % ============================================================
  if ~isequal(fcorr, 'off')
    if nargout == 3
      if isempty(yt)
        lpyt=[];
      else
        lpyt = gp.lik.fh.predy(gp.lik, fvecm', pc_predm', yt, zt);
      end
    elseif nargout > 3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, fvecm', pc_predm', yt, zt);
    end
  else
    if nargout == 3
      if isempty(yt)
        lpyt=[];
      else
        lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
      end
    elseif nargout > 3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
    end
  end
  
end
