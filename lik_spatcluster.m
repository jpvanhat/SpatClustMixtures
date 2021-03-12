function lik = lik_spatcluster(varargin)
%LIK_SPATCLUSTER  Create a multinomial likelihood structure 
%
%  Description
%    LIK = LIK_SPATCLUSTER creates multinomial likelihood for multi-class
%    count data. The observed numbers in each class with C classes is
%    given as 1xC vector.
%
%    The likelihood is defined as follows:
%                              __ n                __ C             
%      p(y|f^1, ..., f^C, z) = || i=1 [ gamma(N+1) || c=1 p_i^c^(y_i^c)/gamma(y_i^c+1)]
%
%    where p_i^c = exp(f_i^c)/ (sum_c=1^C exp(f_i^c)) is the succes 
%    probability for class c, which is a function of the latent variable 
%    f_i^c for the corresponding class and N=sum(y) is the number of trials.
%
%   The parameters are
%    'Snum'
%    'RCPnum'
%    'phi'           
%    'phi_prior'     
%
%
%
%    'alpha'         1xSnum vector 
%    'tau'           1x(RCPnum-1) vector (used as reshape(... )
%    'alpha_prior'   prior
%    'tau_prior'     1x(RCPnum-1) vector (used as reshape(... )
%
%  See also
%    GP_SET, LIK_*
%
% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2010, 2016 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_SPATCLUSTER';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('Snum', [], @(x) isscalar(x) && x>0);
  ip.addParamValue('RCPnum', [], @(x) isscalar(x) && x>0);
  ip.addParamValue('phi', [], @(x) isvector(x) && isreal(x));
  ip.addParamValue('phi_prior', prior_gaussian('s2', 10), @(x) isstruct(x) || isempty(x));
%   ip.addParamValue('alpha', [], @(x) isvector(x) && isreal(x));
%   ip.addParamValue('alpha_prior', prior_gaussian, @(x) isstruct(x) || isempty(x));
%   ip.addParamValue('tau', [], @(x) isvector(x) && isreal(x));
%   ip.addParamValue('tau_prior', prior_gaussian, @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'spatcluster';
    lik.nondiagW = true;
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'spatcluster')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('Snum',ip.UsingDefaults)
      if isempty(ip.Results.Snum)
          error('We need the number of species Snum')
      end
      lik.Snum = ip.Results.Snum;
  end
  if init || ~ismember('RCPnum',ip.UsingDefaults)
      if isempty(ip.Results.RCPnum)
          error('We need the number of regions of common profile (RCP)')
      end
      lik.RCPnum = ip.Results.RCPnum;
  end
  if init || ~ismember('phi',ip.UsingDefaults)
      if isempty(ip.Results.phi)
          lik.phi = zeros(1,lik.Snum.*lik.RCPnum);
      else
          if length(ip.Results.phi) ~= lik.Snum*lik.RCPnum
              error('we need RCPnum phis per species')
          end
          lik.phi = ip.Results.phi;
      end
  end
%   if init || ~ismember('alpha',ip.UsingDefaults)
%       if isempty(ip.Results.alpha)
%           lik.alpha = zeros(1,lik.Snum);
%       else
%           if length(ip.Results.alpha) ~= lik.Snum
%               error('we need one alpha per species')
%           end
%           lik.alpha = ip.Results.alpha;
%       end
%   end
%     if init || ~ismember('tau',ip.UsingDefaults)
%       if isempty(ip.Results.tau)
%           lik.tau = zeros(1,lik.Snum.*(lik.RCPnum-1));
%       else
%           if length(ip.Results.tau) ~= lik.Snum*(lik.RCPnum-1)
%               error('we need RCPnum-1 taus per species')
%           end
%           lik.tau = ip.Results.tau;
%       end
%     end
  
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('phi_prior',ip.UsingDefaults)
    lik.p.phi = ip.Results.phi_prior;
  end
%   if init || ~ismember('alpha_prior',ip.UsingDefaults)
%     lik.p.alpha=ip.Results.alpha_prior;
%   end
%   if init || ~ismember('tau_prior',ip.UsingDefaults)
%     lik.p.tau=ip.Results.tau_prior;
%   end
  
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_spatcluster_pak;
    lik.fh.unpak = @lik_spatcluster_unpak;
    lik.fh.lp = @lik_spatcluster_lp;
    lik.fh.lpg = @lik_spatcluster_lpg;
    lik.fh.ll = @lik_spatcluster_ll;
    lik.fh.llg = @lik_spatcluster_llg;    
    lik.fh.llg2 = @lik_spatcluster_llg2;
    lik.fh.llg3 = @lik_spatcluster_llg3;
    lik.fh.tiltedMoments = @lik_spatcluster_tiltedMoments;
    lik.fh.predy = @lik_spatcluster_predy;
    lik.fh.invlink = @lik_spatcluster_invlink;
    lik.fh.recappend = @lik_spatcluster_recappend;
  end

end  

function [w,s,h] = lik_spatcluster_pak(lik)
%LIK_SPATCLUSTER_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_SPATCLUSTER_PAK(LIK) takes a likelihood structure LIK and
%    returns an empty verctor W. If Multinom likelihood had
%    parameters this would combine them into a single row vector
%    W (see e.g. lik_negbin). This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%     
%
%  See also
%    LIK_SPATCLUSTER_UNPAK, GP_PAK
  
  w=[];s={}; h=[];
  if ~isempty(lik.p.phi)
    w = [w lik.phi];
    s = [s; sprintf('log(lik.phi x %d)',numel(lik.phi))];
    h = [h 0];
    [wh,sh,hh] = lik.p.phi.fh.pak(lik.p.phi);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end
  
%   if ~isempty(lik.p.alpha)
%     w = lik.alpha;
%     s = [s; sprintf('log(lik.alpha x %d)',numel(lik.alpha))];
%     h = [h 0];
%     [wh,sh,hh] = lik.p.alpha.fh.pak(lik.p.alpha);
%     w = [w wh];
%     s = [s; sh];
%     h = [h hh];
%   end
%    if ~isempty(lik.p.tau)
%     w = [w lik.tau];
%     s = [s; sprintf('log(lik.tau x %d)',numel(lik.tau))];
%     h = [h 0];
%     [wh,sh,hh] = lik.p.tau.fh.pak(lik.p.tau);
%     w = [w wh];
%     s = [s; sh];
%     h = [h hh];
%   end
end


function [lik, w] = lik_spatcluster_unpak(lik, w)
%LIK_SPATCLUSTER_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_SPATCLUSTER_UNPAK(W, LIK) Doesn't do anything.
% 
%    If Multinom likelihood had parameters this would extracts them
%    parameters from the vector W to the LIK structure. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%     
%
%  See also
%    LIK_SPATCLUSTER_PAK, GP_UNPAK

if ~isempty(lik.p.phi)
    i1=1;
    i2=length(lik.phi);
    lik.phi = w(i1:i2);
    w = w(i2+1:end);
    % Hyperparameters of phi
    [p, w] = lik.p.phi.fh.unpak(lik.p.phi, w);
    lik.p.phi = p;
end

% if ~isempty(lik.p.alpha)
%     i1=1;
%     i2=length(lik.alpha);
%     lik.alpha = w(i1:i2);
%     w = w(i2+1:end);
%     % Hyperparameters of alpha
%     [p, w] = lik.p.alpha.fh.unpak(lik.p.alpha, w);
%     lik.p.alpha = p;
% end
% if ~isempty(lik.p.tau)
%     i1=1;
%     i2=length(lik.tau);
%     lik.tau = w(i1:i2);
%     w = w(i2+1:end);
%     % Hyperparameters of tau
%     [p, w] = lik.p.tau.fh.unpak(lik.p.tau, w);
%     lik.p.tau = p;
% end

end



function lp = lik_spatcluster_lp(lik, varargin)
%LIK_SPATCLUSTER_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_NEGBIN_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This
%    subfunction is needed if there are likelihood parameters.
%
%  See also
%    LIK_NEGBIN_LLG, LIK_NEGBIN_LLG3, LIK_NEGBIN_LLG2, GPLA_E
  

% If prior for RCP profile parameter, add its contribution
  lp=0;
   if ~isempty(lik.p.phi)
    lp = lp + lik.p.phi.fh.lp(lik.phi, lik.p.phi);
  end
  
%   if ~isempty(lik.p.alpha)
%     lp = lp + lik.p.alpha.fh.lp(lik.alpha, lik.p.alpha);
%   end
%   if ~isempty(lik.p.tau)
%     lp = lp + lik.p.tau.fh.lp(lik.tau, lik.p.tau);
%   end
  
end


function lpg = lik_spatcluster_lpg(lik)
%LIK_SPATCLUSTER_LPG  d log(prior)/dth of the likelihood 
%                parameters th
%
%  Description
%    E = LIK_NEGBIN_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth, where th collects the parameters.
%    This subfunction is needed if there are likelihood parameters.
%
%  See also
%    LIK_NEGBIN_LLG, LIK_NEGBIN_LLG3, LIK_NEGBIN_LLG2, GPLA_G
  
  lpg = [];
  if ~isempty(lik.p.phi)
      lpgs = lik.p.phi.fh.lpg(lik.phi, lik.p.phi);
      lpg = [lpg lpgs];
  end
  
%   if ~isempty(lik.p.alpha)
%       lpgs = lik.p.alpha.fh.lpg(lik.alpha, lik.p.alpha);
%       lpg = [lpg lpgs];
%   end
%   if ~isempty(lik.p.tau)
%       lpgs = lik.p.tau.fh.lpg(lik.tau, lik.p.tau);
%       lpg = [lpg lpgs];
%   end  
  
end  



function [ll, PyGrcp, Prcp, Py] = lik_spatcluster_ll(lik, y, f, z)
%LIK_SPATCLUSTER_LL  Log likelihood
%
%  Description
%    LL = LIK_SPATCLUSTER_LL(LIK, Y, F) takes a likelihood structure
%    LIK, class counts Y (NxC matrix), and latent values F (NxC
%    matrix). Returns the log likelihood, log p(y|f,z). This 
%    subfunction is needed when using Laplace approximation or 
%    MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria 
%    (DIC, WAIC) computations.
%
%  See also
%    LIK_SPATCLUSTER_LLG, LIK_SPATCLUSTER_LLG3, LIK_SPATCLUSTER_LLG2, GPLA_E

  n = size(y,1);
  
  % y is n x Snum
  % f is n*RCPnum
  % pS is RCPnum x Snum
  % pRCP is n x RCPnum
  phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
  pS = 1./(1+exp( -phi ));
  
  f=reshape(f,n,lik.RCPnum);
  expf = exp(f);
  Prcp = expf ./ repmat(sum(expf,2),1,size(expf,2));

  PyGrcp = exp(y*log(pS)' + (1-y)*log(1-pS)');  % pdf of y per site and per RCP
  Py = sum(Prcp.*PyGrcp,2);                     % pdf of y per site
  ll = sum(log(Py));                            % log likelihood

%   PyGrcp = zeros(n,lik.RCPnum);
%   for i1 = 1:lik.RCPnum
%       %PyGrcp(:,i1) = pRCP(:,i1).*prod(phi.^y.*(1-phi).^(1-y),2)
%       phi = repmat(pS(i1,:),n,1);
%       PyGrcp(:,i1) = exp(sum( y.*log(phi) + (1-y).*log(1-phi),2));
%   end
%   ll = sum(log(sum(pRCP.*PyGrcp,2)));
end


function llg = lik_spatcluster_llg(lik, y, f, param, z)
%LIK_SPATCLUSTER_LLG    Gradient of the log likelihood
%
%  Description
%    LLG = LIK_SPATCLUSTER_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the gradient of the log likelihood with respect to PARAM. At
%    the moment PARAM can be 'param' or 'latent'. This subfunction 
%    is needed when using Laplace approximation or MCMC for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_SPATCLUSTER_LL, LIK_SPATCLUSTER_LLG2, LIK_SPATCLUSTER_LLG3, GPLA_E

[~, PyGrcp, Prcp, Py] = lik_spatcluster_ll(lik, y, f);
PyGrcpDPy = bsxfun(@rdivide,PyGrcp,Py);
switch param
    case 'param'
        phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
        pS = 1./(1+exp( -phi ));
        dphi = exp( -phi ).*pS.^2;
        
%         n = size(y,1);
%         llg = zeros(1,lik.RCPnum*lik.Snum);
%         for i1=1:lik.RCPnum
%             phi = repmat(pS(i1,:),n,1);
%             dphi_i1 = repmat(dphi(i1,:),n,1);
%             llill = repmat(PyGrcp(:,i1).*Prcp(:,i1)./Py,1,lik.Snum);
%             %llg = [llg sum( llill./(phi.^y.*(1-phi).^(1-y)).*(y.*phi.^(y-1).*(1-phi).^(1-y) - phi.^y.*(1-y).*(1-phi).^(-y) ).*dphi_i1 ,1)];
%             llg(1+(i1-1)*lik.Snum:i1*lik.Snum) = sum( llill.*(y.*phi.^(-1)-(1-y).*(1-phi).^(-1) ).*dphi_i1 ,1);
%         end
        PyGrcpDPyMPrcp = PyGrcpDPy.*Prcp;
        llg = (  ( (PyGrcpDPyMPrcp'*y)./pS-(PyGrcpDPyMPrcp'*(1-y))./(1-pS) ).*dphi  )';
        llg=llg(:)';

    case 'latent'
        llg = Prcp(:) .* (PyGrcpDPy(:) - 1);
end

end


function [pi_vec, pi_mat] = lik_spatcluster_llg2(lik, y, f, param, z)
%LIK_SPATCLUSTER_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_SPATCLUSTER_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the Hessian of the log likelihood with respect to PARAM. At
%    the moment PARAM can be only 'latent'. LLG2 is a vector with
%    diagonal elements of the Hessian matrix (off diagonals are
%    zero). This subfunction is needed when using Laplace 
%    approximation or EP for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_SPATCLUSTER_LL, LIK_SPATCLUSTER_LLG, LIK_SPATCLUSTER_LLG3, GPLA_E

[~, PyGrcp, Prcp, Py] = lik_spatcluster_ll(lik, y, f);
PyGrcpDPy = bsxfun(@rdivide,PyGrcp,Py);
n = size(y,1);
switch param
    case 'param'
        
    case 'latent'
        llg = Prcp(:) .* (PyGrcpDPy(:) - 1);
        pi_mat = sparse(1:lik.RCPnum*n,repmat(1:n,1,lik.RCPnum),Prcp(:).*PyGrcpDPy(:));
        pi_mat2 = sparse(1:lik.RCPnum*n,repmat(1:n,1,lik.RCPnum),Prcp(:));
        pi_vec = sparse(1:lik.RCPnum*n,1:lik.RCPnum*n,llg) - pi_mat*pi_mat' + pi_mat2*pi_mat2';
        
%         if any(f>0)
%             D = diag(Prcp(:));
%             S = diag(PyGrcpDPy(:));
%             PI = pi_mat2;
%             %tmp = S*D-D-S*PI*PI'*S+PI*PI';
%             %tmp =  (S-eye(size(S)))*D - S*PI*PI'*S+PI*PI';
%             tmp =  (S-eye(size(S)))*D + (eye(size(S))-S)*PI*PI'*(S+eye(size(S)))  ;
%             tmp2 = tmp - PI*PI'*S + S*PI*PI';
%             
%             
%             SD = S*D;
%             SPI = S*PI;
%             
%             tmp2 = (S-eye(size(S)))*D - SPI*SPI' + PI*PI';
%             
%             [V,Lampda]= eig(- SPI*SPI' + PI*PI');
%             plot(diag(Lampda))
%             
%             [V2,Lampda2]= eig((eye(size(S))-S)*PI*PI'*(S+eye(size(S))));
%             hold on,plot(diag(Lampda2))
%             
%             [V,Lampda,U]= svd((eye(size(S))-S)*PI*PI'*(S+eye(size(S))));
%             
%             
%             max(max( tmp2  - pi_vec))
%             min(min( tmp2  - pi_vec))
%             
%             a = randn(size(S,1),1);
%             a'*tmp*a
%             a'*tmp2*a            
%             det(tmp)
%             det(tmp2)
%         end
        
    case 'latent+param'
        phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
        pS = 1./(1+exp( -phi ));
        dphi = exp( -phi ).*pS.^2;

        PyGrcpDPyMPrcp = PyGrcpDPy.*Prcp;
        llg = zeros(n*lik.RCPnum, lik.Snum);
        for i1=1:lik.RCPnum
            PyGrcpDPyMPrcp_i1 = PyGrcpDPyMPrcp(:,i1);
            for j1=1:lik.RCPnum
                tmp = bsxfun(@rdivide,y,pS(j1,:))-bsxfun(@rdivide,1-y,1-pS(j1,:));
                if j1==i1
                    llg((i1-1)*n+1:i1*n,(j1-1)*lik.Snum+1:j1*lik.Snum) = ((1-PyGrcpDPyMPrcp(:,j1)).*PyGrcpDPyMPrcp_i1)*dphi(j1,:).*tmp;
                else
                    llg((i1-1)*n+1:i1*n,(j1-1)*lik.Snum+1:j1*lik.Snum) = (-PyGrcpDPyMPrcp(:,j1).*PyGrcpDPyMPrcp_i1)*dphi(j1,:).*tmp;
                end
            end
        end
        pi_vec = llg;

end

end

function [dw_mat,dw_matIndex] = lik_spatcluster_llg3(lik, y, f, param, z)
%LIK_SPATCLUSTER_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_SPATCLUSTER_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F and
%    returns the third gradients of the log likelihood with
%    respect to PARAM. At the moment PARAM can be only 'latent'. 
%    LLG3 is a vector with third gradients. This subfunction is 
%    needed when using Laplace approximation for inference with 
%    non-Gaussian likelihoods.
%
%  See also
%    LIK_SPATCLUSTER_LL, LIK_SPATCLUSTER_LLG, LIK_SPATCLUSTER_LLG2, GPLA_E, GPLA_G

[~, PyGrcp, Prcp, Py] = lik_spatcluster_ll(lik, y, f);
PyGrcpDPy = bsxfun(@rdivide,PyGrcp,Py);

n = size(y,1);
switch param
    case 'param'
        
    case 'latent'
%         llg = Prcp .* (PyGrcpDPy - 1);
%         
%         %dw_mat={};
%         nout = lik.RCPnum;
%         dw_mat = zeros(nout^3*n,2);
%         dw_mat_vec = zeros(1,lik.RCPnum.^2);
%         iind = zeros(1,lik.RCPnum.^2);
%         jind = zeros(1,lik.RCPnum.^2);
%         for ii1=1:n
%             for cc3=1:lik.RCPnum
%                 counter = 1;
%                 for cc1=1:lik.RCPnum
%                     for cc2=1:lik.RCPnum
%                         % third derivatives
%                         cc_sum_tmp=0;
%                         if cc1==cc2 && cc1==cc3 && cc2==cc3   % all equal
%                             cc_sum_tmp=cc_sum_tmp + llg(ii1,cc1);
%                         end
%                         if cc1==cc2
%                             cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc3).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc3)./(Py(ii1).^2));
%                         end
%                         if cc2==cc3
%                             cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc2).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2)./(Py(ii1).^2));
%                         end
%                         if cc1==cc3
%                             cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc2).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2)./(Py(ii1).^2));
%                         end
%                         cc_sum_tmp=cc_sum_tmp - 2*Prcp(ii1,cc1).*Prcp(ii1,cc2).*Prcp(ii1,cc3).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2).*PyGrcp(ii1,cc3)./(Py(ii1).^3));
%                                                 
%                         iind(counter) = ii1+(cc1-1)*n;
%                         jind(counter) = ii1+(cc2-1)*n;
%                         dw_mat_vec(counter) = cc_sum_tmp;
%                         counter = counter+1;
%                     end
%                 end
%                 dw_mat(1+((ii1-1)+(cc3-1)*n)*nout^2:(ii1+(cc3-1)*n)*nout^2,:) = [(iind+nout*n*(jind-1))' dw_mat_vec'];
%                 %dw_mat{ii1+(cc3-1)*n} = sparse(iind, jind, dw_mat_vec, n*lik.RCPnum,n*lik.RCPnum);
%             end
%         end
        
        llg = Prcp .* (PyGrcpDPy - 1);
        
        nout = lik.RCPnum;
        dw_mat = zeros(nout^3*n,2);
        dw_mat_vec = zeros(lik.RCPnum.^2,n);
        iind = zeros(lik.RCPnum.^2,n);
        jind = zeros(lik.RCPnum.^2,n);
        ii1=1:n;
        for cc3=1:lik.RCPnum
            counter = 1;
            for cc1=1:lik.RCPnum
                for cc2=1:lik.RCPnum
                    % third derivatives
                    cc_sum_tmp=0;
                    if cc1==cc2 && cc1==cc3 && cc2==cc3   % all equal
                        cc_sum_tmp=cc_sum_tmp + llg(ii1,cc1);
                    end
                    if cc1==cc2
                        cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc3).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc3)./(Py(ii1).^2));
                    end
                    if cc2==cc3
                        cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc2).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2)./(Py(ii1).^2));
                    end
                    if cc1==cc3
                        cc_sum_tmp=cc_sum_tmp + Prcp(ii1,cc1).*Prcp(ii1,cc2).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2)./(Py(ii1).^2));
                    end
                    cc_sum_tmp=cc_sum_tmp - 2*Prcp(ii1,cc1).*Prcp(ii1,cc2).*Prcp(ii1,cc3).*(1-PyGrcp(ii1,cc1).*PyGrcp(ii1,cc2).*PyGrcp(ii1,cc3)./(Py(ii1).^3));
                    
                    iind(counter,:) = ii1+(cc1-1)*n;
                    jind(counter,:) = ii1+(cc2-1)*n;
                    dw_mat_vec(counter,:) = cc_sum_tmp;
                    counter = counter+1;
                end
            end
            %inds = (iind+nout*n*(jind-1));
            %dw_mat(1+((cc3-1)*n)*nout^2:(n+(cc3-1)*n)*nout^2,:) = [inds(:) dw_mat_vec(:)];
             dw_mat(1+((cc3-1)*n)*nout^2:(n+(cc3-1)*n)*nout^2,:) = [(iind(:)+nout*n*(jind(:)-1)) dw_mat_vec(:)];
        end
        
        
        
    case 'latent2+param'
%          phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
%         pS = 1./(1+exp( -phi ));
%         dphi = exp( -phi ).*pS.^2;   
%         
%         
%         nout = lik.RCPnum;
%         dw_mat = zeros(n*nout^2,2,lik.RCPnum*lik.Snum);
%         PyGrcpDPyMPrcp = PyGrcpDPy.*Prcp;
%         for cc3=1:lik.RCPnum
%             for k1=1:lik.Snum
%                 dw_mat_vec = zeros(1,n*lik.RCPnum.^2);
%                 iind = zeros(1,n*lik.RCPnum.^2);
%                 jind = zeros(1,n*lik.RCPnum.^2);
%                 counter = 0;
%                 
%                 llill_Dcc3 = PyGrcpDPyMPrcp(:,cc3).*(y(:,k1)./pS(cc3,k1)-(1-y(:,k1))./(1-pS(cc3,k1)));
%                 for i1=1:lik.RCPnum
%                     llill_i1 = PyGrcpDPyMPrcp(:,i1);
%                     for j1=1:lik.RCPnum
%                         llill_j1 = PyGrcpDPyMPrcp(:,j1);
%                         
%                         diag_part = 2*llill_i1.*llill_j1.*llill_Dcc3;
%                         if j1==i1 && j1==cc3
%                             diag_part = diag_part + llill_Dcc3;
%                         end
%                         if j1==i1
%                             diag_part = diag_part  - llill_i1.*llill_Dcc3;
%                         end
%                         if i1==cc3
%                             diag_part = diag_part - llill_j1.*llill_Dcc3;
%                         end
%                         if j1==cc3
%                             diag_part = diag_part - llill_i1.*llill_Dcc3;
%                         end
%                         dw_mat_vec(counter*n+1:(counter+1)*n) = diag_part.*dphi(cc3,k1);
%                         iind(counter*n+1:(counter+1)*n) = 1+(i1-1)*n:i1*n;
%                         jind(counter*n+1:(counter+1)*n) = 1+(j1-1)*n:j1*n;
%                         counter = counter+1;
%                     end
%                 end
%                 %dw_mat2{k1+(cc3-1)*lik.Snum} = [(iind+nout*n*(jind-1))' dw_mat_vec'];
%                 dw_mat(:,:,k1+(cc3-1)*lik.Snum) = [(iind+nout*n*(jind-1))' dw_mat_vec'];
%             end
%         end
        
        
        phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
        pS = 1./(1+exp( -phi ));
        dphi = exp( -phi ).*pS.^2;   
        
        nout = lik.RCPnum;
        dw_mat = zeros(n*nout^2,lik.RCPnum*lik.Snum);
        dw_matIndex = zeros(n*nout^2,1);
        PyGrcpDPyMPrcp = PyGrcpDPy.*Prcp;
        for cc3=1:lik.RCPnum
            k1=1:lik.Snum;
            counter = 0;
            
            llill_Dcc3 = PyGrcpDPyMPrcp(:,cc3).*(y(:,k1)./pS(cc3,k1)-(1-y(:,k1))./(1-pS(cc3,k1)));
            for i1=1:lik.RCPnum
                llill_i1 = PyGrcpDPyMPrcp(:,i1);
                for j1=1:lik.RCPnum
                    llill_j1 = PyGrcpDPyMPrcp(:,j1);
                    
                    diag_part = 2*(llill_i1.*llill_j1).*llill_Dcc3;
                    if j1==i1 && j1==cc3
                        diag_part = diag_part + llill_Dcc3;
                    end
                    if j1==i1
                        diag_part = diag_part  - llill_i1.*llill_Dcc3;
                    end
                    if i1==cc3
                        diag_part = diag_part - llill_j1.*llill_Dcc3;
                    end
                    if j1==cc3
                        diag_part = diag_part - llill_i1.*llill_Dcc3;
                    end
%                     iind = 1+(i1-1)*n:i1*n;
%                     jind = 1+(j1-1)*n:j1*n;
%                     dw_mat(counter*n+1:(counter+1)*n,1,k1+(cc3-1)*lik.Snum) = repmat((iind+nout*n*(jind-1))',1,100);
%                     dw_mat(counter*n+1:(counter+1)*n,2,k1+(cc3-1)*lik.Snum) = diag_part.*dphi(cc3,k1);
                    
                    if cc3==1
                        iind = 1+(i1-1)*n:i1*n;
                        jind = 1+(j1-1)*n:j1*n;
                        dw_matIndex(counter*n+1:(counter+1)*n) = (iind+nout*n*(jind-1))';
                    end
                    dw_mat(counter*n+1:(counter+1)*n,k1+(cc3-1)*lik.Snum) = diag_part.*dphi(cc3,k1);

                    counter = counter+1;
                end
                a=1;
            end            
        end
        
        
%         phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
%         pS = 1./(1+exp( -phi ));
%         dphi = exp( -phi ).*pS.^2;   
%         
%         
%         nout = lik.RCPnum;
% %         dw_mat = zeros((nout*n)^2,2);
%         PyGrcpDPyMPrcp = PyGrcpDPy.*Prcp;
%         for cc3=1:lik.RCPnum
%             for k1=1:lik.Snum
%                 dw_mat_vec = zeros(1,n*lik.RCPnum.^2);
%                 iind = zeros(1,n*lik.RCPnum.^2);
%                 jind = zeros(1,n*lik.RCPnum.^2);
%                 counter = 0;
%                 
%                 llill_Dcc3 = PyGrcpDPyMPrcp(:,cc3).*(y(:,k1)./pS(cc3,k1)-(1-y(:,k1))./(1-pS(cc3,k1)));
%                 for i1=1:lik.RCPnum
%                     llill_i1 = PyGrcpDPyMPrcp(:,i1);
%                     for j1=1:lik.RCPnum
%                         llill_j1 = PyGrcpDPyMPrcp(:,j1);
%                         
%                         diag_part = 2*llill_i1.*llill_j1.*llill_Dcc3;
%                         if j1==i1 && j1==cc3
%                             diag_part = diag_part + llill_Dcc3;
%                         end
%                         if j1==i1
%                             diag_part = diag_part  - llill_i1.*llill_Dcc3;
%                         end
%                         if i1==cc3
%                             diag_part = diag_part - llill_j1.*llill_Dcc3;
%                         end
%                         if j1==cc3
%                             diag_part = diag_part - llill_i1.*llill_Dcc3;
%                         end
%                         dw_mat_vec(counter*n+1:(counter+1)*n) = diag_part.*dphi(cc3,k1);
%                         iind(counter*n+1:(counter+1)*n) = 1+(i1-1)*n:i1*n;
%                         jind(counter*n+1:(counter+1)*n) = 1+(j1-1)*n:j1*n;
%                         counter = counter+1;
%                     end
%                 end
%                 %dw_mat(1+((ii1-1)+(cc3-1)*n)*nout^2:(ii1+(cc3-1)*n)*nout^2,:) = [(iind+nout*n*(jind-1))' dw_mat_vec'];
%                 %dw_mat{k1+(cc3-1)*lik.Snum} = sparse(iind,jind,dw_mat_vec,n*lik.RCPnum,n*lik.RCPnum);
%                 dw_mat{k1+(cc3-1)*lik.Snum} = [(iind+nout*n*(jind-1))' dw_mat_vec'];
%             end
%         end
        


        % original
%         dw_mat = zeros(n*lik.RCPnum, n*lik.RCPnum, lik.Snum);
%         for cc3=1:lik.RCPnum
%             for k1=1:lik.Snum
%                 dw_mat_tmp = [];
%                 phi = repmat(pS(cc3,k1),n,1);
%                 dphi_cc3 = dphi(cc3,k1);
%                 llill_Dcc3 = PyGrcp(:,cc3).*Prcp(:,cc3)./Py.*(y(:,k1).*phi.^(-1)-(1-y(:,k1)).*(1-phi).^(-1));
%                 for i1=1:lik.RCPnum
%                     llill_i1 = PyGrcp(:,i1).*Prcp(:,i1)./Py;
%                     for j1=1:lik.RCPnum
%                         llill_j1 = PyGrcp(:,j1).*Prcp(:,j1)./Py;
%                         
%                         diag_part = 2*llill_i1.*llill_j1.*llill_Dcc3;
%                         if j1==i1 && j1==cc3
%                             diag_part = diag_part + llill_Dcc3;
%                         end
%                         if j1==i1
%                             diag_part = diag_part  - llill_i1.*llill_Dcc3;
%                         end
%                         if i1==cc3
%                             diag_part = diag_part - llill_j1.*llill_Dcc3;
%                         end
%                         if j1==cc3
%                             diag_part = diag_part - llill_i1.*llill_Dcc3;
%                         end                        
%                         dw_mat(1+(i1-1)*n:i1*n,1+(j1-1)*n:j1*n,k1+(cc3-1)*lik.Snum) = diag(diag_part).*dphi_cc3;
%                     end
%                 end
%             end
%         end
end
end

function [logM_0, m_1, sigm2hati1] = lik_spatcluster_tiltedMoments(lik, y, i1, S2_i, M_i, z)
    %LIK_COXPH_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
    %
    %  Description
    %    [M_0, M_1, M2] = LIK_COXPH_TILTEDMOMENTS(LIK, Y, I, S2,
    %    MYY, Z) takes a likelihood structure LIK, class labels
    %    Y, index I and cavity variance S2 and
    %    mean MYY. Returns the zeroth moment M_0, mean M_1 and
    %    variance M_2 of the posterior marginal (see Rasmussen and
    %    Williams (2006): Gaussian processes for Machine Learning,
    %    page 55). This subfunction is needed when using EP for 
    %    inference with non-Gaussian likelihoods.
    %
    %  See also
    %    GPEP_E
    
    error('tiltedMoment has not been implemented ');
    
end

function [lpy, Ey, Vary] = lik_spatcluster_predy(lik, Ef, Varf, yt, zt)
%LIK_SPATCLUSTER_PREDY  Returns the predictive mean, variance and density of y
%
%  Description
%    LPY = LIK_SPATCLUSTER_PREDY(LIK, EF, VARF YT)
%    Returns logarithm of the predictive density PY of YT, that is
%        p(yt | y) = \int p(yt | f) p(f|y) df.
%    This requires also the incedence counts YT. This subfunction 
%    is needed when computing posterior predictive distributions for 
%    future observations.
%
%    [LPY, EY, VARY] = LIK_SPATCLUSTER_PREDY(LIK, EF, VARF, YT) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for
%    future observations.
%

%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED
  

  n = size(yt,1);
  
  % yt is n x Snum
  % f is n*RCPnum
  % pS is RCPnum x Snum
  % pRCP is n x RCPnum
  
  % pdf of yt per site and per RCP
  phi = reshape(lik.phi, lik.Snum, lik.RCPnum)';
  pS = 1./(1+exp( -phi ));
  PyGrcp = exp(yt*log(pS)' + (1-yt)*log(1-pS)');
  
  % calculate the predictive probabilities for RCPs
  S=10000;
  [ntest,nout]=size(Ef);
  lpy=zeros(ntest,1);
  for i1=1:ntest
      Sigm_tmp=(Varf(:,:,i1)'+Varf(:,:,i1))./2;
      f_star=mvnrnd(Ef(i1,:), Sigm_tmp, S);
      
      Prcp = exp(f_star);
      Prcp = Prcp./(sum(Prcp, 2)*ones(1,size(Prcp,2)));
      % pdf of yt per site
      lpy(i1) = log(mean(sum(Prcp.*PyGrcp(i1,:),2)));
  end
%   f=reshape(Ef,n,lik.RCPnum);
%   expf = exp(f);
%   Prcp = expf ./ repmat(sum(expf,2),1,size(expf,2));
%   % log probability density per site 
%   lpy = log( sum(Prcp.*PyGrcp,2) );
  
  if nargout > 1
      error('predy has not been implemented for EY and VarY');
  end
end

function p = lik_spatcluster_invlink(lik, f, z)
%LIK_SPATCLUSTER_INVLINK Returns values of inverse link function
%             
%  Description 
%    P = LIK_SPATCLUSTER_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values of inverse link function P.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_SPATCLUSTER_LL, LIK_SPATCLUSTER_PREDY
 error('invlink has not been implemented ');
end

function reclik = lik_spatcluster_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description
%    RECLIK = LIK_SPATCLUSTER_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction
%    is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC

if nargin == 2
    reclik.type = 'spatcluster';
    reclik.nondiagW = true;
    reclik.Snum = ri.Snum;
    reclik.RCPnum  = ri.RCPnum;
    
    reclik.phi = [];
    
    % Set the function handles
    reclik.fh.pak = @lik_spatcluster_pak;
    reclik.fh.unpak = @lik_spatcluster_unpak;
    reclik.fh.lp = @lik_spatcluster_lp;
    reclik.fh.lpg = @lik_spatcluster_lpg;
    reclik.fh.ll = @lik_spatcluster_ll;
    reclik.fh.llg = @lik_spatcluster_llg;
    reclik.fh.llg2 = @lik_spatcluster_llg2;
    reclik.fh.llg3 = @lik_spatcluster_llg3;
    reclik.fh.tiltedMoments = @lik_spatcluster_tiltedMoments;
    reclik.fh.predy = @lik_spatcluster_predy;
    reclik.fh.invlink = @lik_spatcluster_invlink;
    reclik.fh.recappend = @lik_spatcluster_recappend;
    
    reclik.p=[];
    reclik.p.phi=[];
    if ~isempty(ri.p.phi)
      reclik.p.phi = ri.p.phi;
    end
    
else
    % Append to the record
    reclik.phi(ri,:)=lik.phi;
    if ~isempty(lik.p.phi)
        reclik.p.phi = lik.p.phi.fh.recappend(reclik.p.phi, ri, lik.p.phi);
    end
    
end
  
end




% code for checking the implementation of gradients llg*
% 
% lik = lik_spatcluster('Snum', 4, 'RCPnum', 3)
% 
% 
% [w,s] = lik.fh.pak(lik)
% w=randn(size(w));
% %w=1:length(w)
% lik = lik.fh.unpak(lik,w)
% 
% 
% y = [1 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 1 0];
% f = 2*randn(size(y,1)*lik.RCPnum,1);
% 
% 
% lik.fh.ll(lik,y,f)
% 
% % check llg with respect to latent
% fe = @(x) lik.fh.ll(lik,y,x');
% fg = @(x) lik.fh.llg(lik,y,x','latent')';
% gradcheck(randn(size(f')),fe,fg);
% 
% % check llg with respect to param
% fe = @(x) lik.fh.ll(lik.fh.unpak(lik,x),y,f);
% fg = @(x) lik.fh.llg(lik.fh.unpak(lik,x),y,f,'param');
% gradcheck(randn(size(w)),fe,fg);
% 
% % check llg2 with respect to latent
% h1 = lik.fh.llg2(lik,y,f, 'latent');
% fe = @(x) lik.fh.ll(lik,y,x);
% h2 = hessian(fe,f);
% [min(min(h1)) max(max(h1))]
% [min(min(h2)) max(max(h2))]
% [min(min(h1-h2)) max(max(h1-h2))]
% 
% % check llg2 with respect to latent+param
% take_ith = @(M,i) M(i,:);
% fe = @(x,i) take_ith(lik.fh.llg(lik.fh.unpak(lik,x),y,f,'latent'),i);
% fg = @(x,i) take_ith(lik.fh.llg2(lik.fh.unpak(lik,x),y,f,'latent+param'),i);
% gradcheck(randn(size(w)),fe,fg,randpick(1:length(f)));
% 
% % check llg3 with respect to latent+param
% %  you need to take each element of llg2 at time
% take_ijth = @(M,i,j) squeeze(M(i,j,:))';
% fe = @(x,i,j) take_ijth(full(lik.fh.llg2(lik.fh.unpak(lik,x),y,f,'latent')),i,j);
% fg = @(x,i,j) take_ijth(lik.fh.llg3(lik.fh.unpak(lik,x),y,f,'latent2+param'),i,j);
% gradcheck(randn(size(w)),fe,fg,randpick(1:length(f)),randpick(1:length(f)));
% 
% i1=randpick(1:length(f));j1=randpick(1:length(f));
% %i1=j1;
% [i1,j1]
% gradcheck(randn(size(w)),fe,fg,i1,j1);




