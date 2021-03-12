function [e, edata, eprior, param] = gpla_e(w, gp, varargin)
%GPLA_E  Do Laplace approximation and return marginal log posterior estimate
%
%  Description
%    E = GPLA_E(W, GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors, and finds the Laplace approximation for the
%    conditional posterior p(Y | X, th), where th is the
%    parameters. Returns the energy at th (see below). Each
%    row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPLA_E(W, GP, X, Y, OPTIONS) returns also
%    the data and prior components of the total energy.
%
%    The energy is minus log posterior cost function for th:
%      E = EDATA + EPRIOR
%        = - log p(Y|X, th) - log p(th),
%      where th represents the parameters (lengthScale,
%      magnSigma2...), X is inputs and Y is observations.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  References
%
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%    Jarno Vanhatalo, Pasi Jylänki and Aki Vehtari (2009). Gaussian
%    process regression with Student-t likelihood. In Y. Bengio et al,
%    editors, Advances in Neural Information Processing Systems 22,
%    pp. 1910-1918
%
%  See also
%    GP_SET, GP_E, GPLA_G, GPLA_PRED

%  Description 2
%    Additional properties meant only for internal use.
%
%    GP = GPLA_E('init', GP) takes a GP structure GP and
%    initializes required fields for the Laplace approximation.
%
%    GP = GPLA_E('clearcache', GP) takes a GP structure GP and clears the
%    internal cache stored in the nested function workspace.
%
%    [e, edata, eprior, f, L, a, La2, p] = GPLA_E(w, gp, x, y, varargin)
%    returns many useful quantities produced by EP algorithm.
%
%    The Newton's method is implemented as described in Rasmussen
%    and Williams (2006).
%
%    The stabilized Newton's method is implemented as suggested by
%    Hannes Nickisch (personal communication).
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2010 Pasi Jylänki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GPLA_E';
ip.addRequired('w', @(x) ...
    isempty(x) || ...
    (ischar(x) && strcmp(w, 'init')) || ...
    isvector(x) && isreal(x) && all(isfinite(x)) ...
    || all(isnan(x)));
ip.addRequired('gp',@isstruct);
ip.addOptional('x', @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('y', @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;

if strcmp(w, 'init')
    % Initialize cache
    ch = [];
    
    % set function handle to the nested function laplace_algorithm
    % this way each gp has its own peristent memory for EP
    gp.fh.ne = @laplace_algorithm;
    % set other function handles
    gp.fh.e=@gpla_e;
    gp.fh.g=@gpla_g;
    gp.fh.pred=@gpla_pred;
    gp.fh.jpred=@gpla_jpred;
    gp.fh.looe=@gpla_looe;
    gp.fh.loog=@gpla_loog;
    gp.fh.loopred=@gpla_loopred;
    e = gp;
    % remove clutter from the nested workspace
    clear w gp varargin ip x y z
elseif strcmp(w, 'clearcache')
    % clear the cache
    gp.fh.ne('clearcache');
else
    % call laplace_algorithm using the function handle to the nested function
    % this way each gp has its own peristent memory for Laplace
    %[e, edata, eprior, f, L, a, La2, p] = gp.fh.ne(w, gp, x, y, z);
    [e, edata, eprior, param] = gp.fh.ne(w, gp, x, y, z);
end

    function [e, edata, eprior, param] = laplace_algorithm(w, gp, x, y, z)
        
        if strcmp(w, 'clearcache')
            ch=[];
            return
        end
        % code for the Laplace algorithm
        
        % check whether saved values can be used
        if isempty(z)
            datahash=hash_sha512([x y]);
        else
            datahash=hash_sha512([x y z]);
        end
        if ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1e-8) && ...
                isequal(datahash,ch.datahash)
            % The covariance function parameters or data haven't changed so we
            % can return the energy and the site parameters that are
            % saved in the cache
            e = ch.e;
            edata = ch.edata;
            eprior = ch.eprior;
            param.f = ch.f;
            param.L = ch.L;
            param.La2 = ch.La2;
            param.a = ch.a;
            param.p = ch.p;
        else
            % The parameters or data have changed since
            % the last call for gpla_e. In this case we need to
            % re-evaluate the Laplace approximation
            gp=gp_unpak(gp, w);
            ncf = length(gp.cf);
            n = size(x,1);
            p = [];
            maxiter = gp.latent_opt.maxiter;
            tol = gp.latent_opt.tol;
            
            % Initialize latent values
            % zero seems to be a robust choice (Jarno)
            % with mean functions, initialize to mean function values
            if ~isfield(gp,'meanf')
                f = zeros(size(y));
            else
                [H,b_m,B_m]=mean_prep(gp,x,[]);
                f = H'*b_m;
            end
            
            % =================================================
            % First Evaluate the data contribution to the error
            switch gp.type
                % ============================================================
                % FULL
                % ============================================================
                case 'FULL'
                    
                    if ~isfield(gp.lik, 'nondiagW')
                        

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
                                error('GPLA_ND_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
                            end
                        else
                            multicf = false;
                        end
                        p=[];
                        switch gp.lik.type
                                
                            case {'Softmax', 'Multinom'}
                                
                                % Initialize latent values
                                % zero seems to be a robust choice (Jarno)
                                switch gp.lik.type
                                    case {'Softmax', 'Multinom'}
                                        f = zeros(size(y(:)));
                                    case {'spatcluster'}
                                        nout = gp.lik.RCPnum;
                                        f = zeros(size(y,1).*nout,1);
                                end
                                
                                K = zeros(n,n,nout);
                                if multicf
                                    for i1=1:nout
                                        K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
                                    end
                                else
                                    Ktmp=gp_trcov(gp, x);
                                    for i1=1:nout
                                        K(:,:,i1) = Ktmp;
                                    end
                                end
                                
                                % Main newton algorithm, see Rasmussen & Williams (2006),
                                % p. 50
                                
                                tol = 1e-12;
                                a = f;
                                
                                f2=reshape(f,n,nout);
                                
                                % lp_new = log(p(y|f))
                                lp_new = gp.lik.fh.ll(gp.lik, y, f2, z);
                                lp_old = -Inf;
                                
                                c=zeros(n*nout,1);
                                ERMMRc=zeros(n*nout,1);
                                E=zeros(n,n,nout);
                                L=zeros(n,n,nout);
                                RER = zeros(n,n,nout);
                                
                                while lp_new - lp_old > tol
                                    lp_old = lp_new; a_old = a;
                                    
                                    % llg = d(log(p(y|f)))/df
                                    llg = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
                                    % Second derivatives
                                    [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
                                    % W = -diag(pi2_vec) + pi2_mat*pi2_mat'
                                    pi2 = reshape(pi2_vec,n,nout);
                                    
                                    R = repmat(1./pi2_vec,1,n).*pi2_mat;
                                    for i1=1:nout
                                        Dc=sqrt(pi2(:,i1));
                                        Lc=(Dc*Dc').*K(:,:,i1);
                                        Lc(1:n+1:end)=Lc(1:n+1:end)+1;
                                        [Lc,notpositivedefinite]=chol(Lc);
                                        if notpositivedefinite
                                            [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                            return
                                        end
                                        L(:,:,i1)=Lc;
                                        
                                        Ec=Lc'\diag(Dc);
                                        Ec=Ec'*Ec;
                                        E(:,:,i1)=Ec;
                                        RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
                                    end
                                    [M, notpositivedefinite]=chol(sum(RER,3));
                                    if notpositivedefinite
                                        [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    
                                    b = pi2_vec.*f - pi2_mat*(pi2_mat'*f) + llg;
                                    for i1=1:nout
                                        c((1:n)+(i1-1)*n)=E(:,:,i1)*(K(:,:,i1)*b((1:n)+(i1-1)*n));
                                    end
                                    
                                    RMMRc=R*(M\(M'\(R'*c)));
                                    for i1=1:nout
                                        ERMMRc((1:n)+(i1-1)*n) = E(:,:,i1)*RMMRc((1:n)+(i1-1)*n,:);
                                    end
                                    a=b-c+ERMMRc;
                                    
                                    for i1=1:nout
                                        f((1:n)+(i1-1)*n)=K(:,:,i1)*a((1:n)+(i1-1)*n);
                                    end
                                    f2=reshape(f,n,nout);
                                    
                                    lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f2, z);
                                    
                                    i = 0;
                                    while i < 10 && lp_new < lp_old  || isnan(sum(f))
                                        % reduce step size by half
                                        a = (a_old+a)/2;
                                        
                                        for i1=1:nout
                                            f((1:n)+(i1-1)*n)=K(:,:,i1)*a((1:n)+(i1-1)*n);
                                        end
                                        f2=reshape(f,n,nout);
                                        
                                        lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f2, z);
                                        i = i+1;
                                    end
                                end
                                
                                [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
                                pi2 = reshape(pi2_vec,n,nout);
                                
                                zc=0;
                                Detn=0;
                                R = repmat(1./pi2_vec,1,n).*pi2_mat;
                                for i1=1:nout
                                    Dc=sqrt( pi2(:,i1) );
                                    Lc=(Dc*Dc').*K(:,:,i1);
                                    Lc(1:n+1:end)=Lc(1:n+1:end)+1;
                                    [Lc, notpositivedefinite]=chol(Lc);
                                    if notpositivedefinite
                                        [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    L(:,:,i1)=Lc;
                                    
                                    pi2i = pi2_mat((1:n)+(i1-1)*n,:);
                                    pipi = pi2i'/diag(Dc);
                                    Detn = Detn + pipi*(Lc\(Lc'\diag(Dc)))*K(:,:,i1)*pi2i;
                                    zc = zc + sum(log(diag(Lc)));
                                    
                                    Ec=Lc'\diag(Dc);
                                    Ec=Ec'*Ec;
                                    E(:,:,i1)=Ec;
                                    RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
                                end
                                [M, notpositivedefinite]=chol(sum(RER,3));
                                if notpositivedefinite
                                    [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                    return
                                end
                                
                                zc = zc + sum(log(diag(chol( eye(size(K(:,:,i1))) - Detn))));
                                
                                logZ = a'*f/2 - gp.lik.fh.ll(gp.lik, y, f2, z) + zc;
                                edata = logZ;
                                
                            case 'spatcluster'
                                % Initialize latent values
                                % zero seems to be a robust choice (Jarno)
                                nout = gp.lik.RCPnum;
                                f = zeros(size(y,1).*nout,1);
                                
                                K = sparse(n*nout,n*nout);
                                if multicf
                                    for i1=1:nout
                                        K((i1-1)*n+1:i1*n,(i1-1)*n+1:i1*n) = gp_trcov(gp, x, gp.comp_cf{i1});
                                    end
                                else
                                    Ktmp=gp_trcov(gp, x);
                                    for i1=1:nout
                                        K((i1-1)*n+1:i1*n,(i1-1)*n+1:i1*n) = Ktmp;
                                    end
                                end
                                
                                % --------------------------------------------------------------------------------
                                % find the posterior mode of latent variables by stabilized Newton method.
                                % --------------------------------------------------------------------------------
                                a=f;
                                [LK, notpositivedefinite] = chol(K, 'lower');
                                if notpositivedefinite
                                    [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                    return
                                end
                                lp = -f'*(LK'\(LK\f))/2 +gp.lik.fh.ll(gp.lik, y, f, z);
                                
                                lp_old = -Inf;
                                f_old = f+1;
                                W = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                                %dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z) - LK'\(LK\f);
                                dlp = W*f + gp.lik.fh.llg(gp.lik, y, f, 'latent', z);

                                %B = LK'*(W*LK);  % !! this is full anyways and, hence, sparse matrix will be slower
                                B = LK'*full(W*LK);  % !! this is full anyways and, hence, sparse matrix will be slower
                                B(1:n*nout+1:end)=B(1:n*nout+1:end)+1;
                                [LB, notpositivedefinite] = chol(B, 'lower');

                                iter=0;
                                % begin Newton's iterations
                                while (lp - lp_old > tol || max(abs(f-f_old)) > tol) && iter < maxiter
                                    iter=iter+1;
                                    lp_old = lp; a_old = a; f_old = f;
                                    
                                    if notpositivedefinite    % stabilized-newton iteration
                                        sW = full(sqrt(max(diag(W),0)));
                                        LB=bsxfun(@times,bsxfun(@times,sW,K),sW');
                                        LB(1:n*nout+1:end)=LB(1:n*nout+1:end)+1;
                                        [LB,notpositivedefinite] = chol(LB, 'lower');
                                        b = K*dlp;
                                        delta = b - K*(sW.*(LB'\(LB\(sW.*b))));
                                    else                        % Regular Newton iteration
                                        delta = LK*(LB'\(LB\(LK'*dlp)));
                                        %iter
                                    end
                                    %f = f + delta;
                                    f = delta;
                                    if any(isnan(f))
                                        [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    a = LK'\(LK\f);
                                    lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f, z);
                                    i = 0;
                                    while i < 25 && (lp_new < lp_old  || isnan(sum(f)) || isnan(lp_new))
                                        % reduce step size by half until improvement in
                                        % objective
                                        a = (a_old+a)/2;
                                        f = K*a;
                                        lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f, z);
                                        i = i+1;
                                    end
                                    %if i>=25
                                    %    warning('step size might still be too large in Newton iteration')
                                    %end
                                    lp = lp_new;
                                    W = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                                    %dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z) - LK'\(LK\f);
                                    dlp = W*f + gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
                                                                        
                                    %B = LK'*(W*LK);  % !! this is full anyways and, hence, sparse matrix will be slower
                                    B = LK'*full(W*LK);  % !! this is full anyways and, hence, sparse matrix will be slower
                                    B(1:n*nout+1:end)=B(1:n*nout+1:end)+1;
                                    [LB, notpositivedefinite] = chol(B, 'lower');

                                    %plot(f),pause
                                end
                                                               
                                if iter>=maxiter
                                    warning('Newton iteration may not have converged. You might want to increase maxiter.')
                                end
                                
                                % Calculate the approximate log marginal likelihood
                                edata = -lp + sum(log(diag(LB)));
                                
                                if notpositivedefinite
                                    [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite();
                                    return
                                end
                                E = W;
                                L = LK;
                                M = LB;
                            otherwise
                                
                            
                        end
                        La2=E;
                        p=M;
                        
                    end
                    
                 
                    
                otherwise
                    error('Unknown type of Gaussian process!')
            end
            
            % ======================================================================
            % Evaluate the prior contribution to the error from covariance functions
            % ======================================================================
            eprior = 0;
            for i1=1:ncf
                gpcf = gp.cf{i1};
                eprior = eprior - gpcf.fh.lp(gpcf);
            end
            
            % ======================================================================
            % Evaluate the prior contribution to the error from likelihood function
            % ======================================================================
            if isfield(gp, 'lik') && isfield(gp.lik, 'p')
                lik = gp.lik;
                eprior = eprior - lik.fh.lp(lik);
            end
            
            % ============================================================
            % Evaluate the prior contribution to the error from the inducing inputs
            % ============================================================
            if ~isempty(strfind(gp.infer_params, 'inducing'))
                if isfield(gp, 'p') && isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                    if iscell(gp.p.X_u) % Own prior for each inducing input
                        for i = 1:size(gp.X_u,1)
                            pr = gp.p.X_u{i};
                            eprior = eprior - pr.fh.lp(gp.X_u(i,:), pr);
                        end
                    else
                        eprior = eprior - gp.p.X_u.fh.lp(gp.X_u(:), gp.p.X_u);
                    end
                end
            end
            
            % ============================================================
            % Evaluate the prior contribution to the error from mean functions
            % ============================================================
            if ~isempty(strfind(gp.infer_params, 'mean'))
                for i=1:length(gp.meanf)
                    gpmf = gp.meanf{i};
                    eprior = eprior - gpmf.fh.lp(gpmf);
                end
            end
            
            e = edata + eprior;
            
            % store values to struct param
            param.f = f;
            param.L = L;
            param.La2 = La2;
            param.a = a;
            param.p=p;
            
            % store values to the cache
            ch = param;
            ch.w = w;
            ch.e = e;
            ch.edata = edata;
            ch.eprior = eprior;
            ch.n = size(x,1);
            ch.datahash=datahash;
        end
        
        %    assert(isreal(edata))
        %    assert(isreal(eprior))
        
        %
        % ==============================================================
        % Begin of the nested functions
        % ==============================================================
        %
        function [e, g, h] = egh(f, varargin)
            ikf = iKf(f');
            e = 0.5*f*ikf - gp.lik.fh.ll(gp.lik, y, f', z);
            g = (ikf - gp.lik.fh.llg(gp.lik, y, f', 'latent', z))';
            h = -gp.lik.fh.llg2(gp.lik, y, f', 'latent', z);
        end
        function ikf = iKf(f, varargin)
            
%             switch gp.type
%                 case {'PIC' 'PIC_BLOCK'}
%                     iLaf = zeros(size(f));
%                     for i=1:length(ind)
%                         iLaf(ind2depo{i},:) = LLabl{i}\(LLabl{i}'\f(ind{i},:));
%                     end
%                     ikf = iLaf - L*(L'*f);
%                 case 'CS+FIC'
%                     ikf = ldlsolve(VD,f) - L*(L'*f);
%             end
        end
    end
    function [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite()
        % Instead of stopping to chol error, return NaN
        edata=NaN;
        e=NaN;
        eprior=NaN;
        param.f=NaN;
        param.L=NaN;
        param.a=NaN;
        param.La2=NaN;
        param.p=NaN;
        w=NaN;
        datahash = NaN;
        ch=param;
        ch.e = e;
        ch.edata = edata;
        ch.eprior = eprior;
        ch.datahash=datahash;
        ch.w = NaN;
    end

end
