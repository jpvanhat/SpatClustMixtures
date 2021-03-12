%%

rng(1)

% Define the number of RCP clusters
RCPnum = 3;

% choose whether (true) or not (false) to do 10-fold cross validation
docv = false;

% === Select the RCP initialization method ===
init = 'kmeanInit';
%init = 'default';
%init = 'totalRand';

% === Select the predictor function model ===
%model = 'polynomial';        % The same model as used by Foster et al. (2013)
model = 'matern52';           % The model used by Vanhatalo et al (manuscript)
%model = 'sexp';

% identifiability constraint ====
% ZeroFirstLatent = 'ZeroFirstLatent';  % set the predictors of the first
                                        % RCP to (approximately) zero for identifiability
ZeroFirstLatent = '';                   % let all RCPs-wise predictors vary

% === print figures ===
printFigure = false;      % set this to true if you want to print/save the figures

% Prior for the weights of linear (polynomial) terms
betasigma_fixed = '_bsFixed';       % Gaussian with fixed (large) variance 
%betasigma_fixed = '';              % Gaussian with shrinkage hyperprior
                                    % for the variance parameter

% Load the data
data = importdata('simulated_data/SimulatedData.csv',',',1);
switch model
    case 'polynomial'
        x = data.data(:,101:107) ;
        % Prediction raster
        dataPred = importdata('simulated_data/simDat_predGrid.csv',',',1);
        xpred = dataPred.data(:,[1 4:9]);
        xd = [dataPred.data(:,2),dataPred.data(:,3)];
        %plot(xd(:,1),xd(:,2),'.')
        xd = round((xd - repmat(min(xd),size(xd,1),1))./0.202020202020+1);
        xii=sub2ind([max(xd(:,1)) max(xd(:,2))],xd(:,1),xd(:,2));
        [X1, X2] = meshgrid(1:max(xd(:,1)), 1:max(xd(:,2)));
    case {'matern52' 'sexp'}
        x = [data.data(:,strcmp(data.textdata,'"x1.1"')) data.data(:,strcmp(data.textdata,'"x2.1"'))];
        % construct prediction raster
        %------------------------
        [X1,X2] = meshgrid(linspace(-0.15,0.15,20), linspace(-0.15,0.15,20));
        xpred = [X1(:),X2(:)];
        x=x*10;
        xpred=xpred*10;
end
% put species data into y. 
yind = false(1, length(data.textdata));
for i1=1:length(data.textdata)
    yind(i1) = ~isempty(strfind(data.textdata{i1}, 'spp'));
end
y = double(data.data(:,yind));

lik = lik_spatcluster('Snum', size(y,2), 'RCPnum', RCPnum);
cfmatern52 = gpcf_matern52('lengthScale', [0.5 0.5], 'magnSigma2_prior', prior_sqrtt, 'lengthScale_prior', prior_invt);
cfsexp = gpcf_sexp('lengthScale', [0.5 0.5], 'magnSigma2_prior', prior_sqrtt, 'lengthScale_prior', prior_invt);
if ~isempty(betasigma_fixed)
    cflin = gpcf_linear('coeffSigma2_prior', prior_fixed, 'coeffSigma2', 5); 
    jitterSigma=1e-4;
else
    cflin = gpcf_linear; 
    jitterSigma=1e-4;
end
switch model
    case 'polynomial'
        cf_all = repmat({cflin},1,lik.RCPnum);
        comp_cf = {};
        for i1=1:RCPnum
            comp_cf{end+1} = 1+(i1-1);
        end
        if ~isempty(ZeroFirstLatent)
            cf_all{1}.coeffSigma2 = 0.001;
        end
    case 'matern52'
        cf_all = repmat({cfmatern52},1,lik.RCPnum);
        comp_cf = {};
        for i1=1:RCPnum
            comp_cf{end+1} = 1+(i1-1);
        end
        if ~isempty(ZeroFirstLatent)
            cf_all{1}.p.lengthScale = prior_fixed;
            cf_all{1}.p.magnSigma2 = prior_fixed;
            cf_all{1}.lengthScale = [5 5];
            cf_all{1}.magnSigma2 = 0.00001;
        end
    case 'sexp'
        cf_all = repmat({cfsexp},1,lik.RCPnum);
        comp_cf = {};
        for i1=1:RCPnum
            comp_cf{end+1} = 1+(i1-1);
        end
        if ~isempty(ZeroFirstLatent)
            cf_all{1}.p.lengthScale = prior_fixed;
            cf_all{1}.p.magnSigma2 = prior_fixed;
            cf_all{1}.lengthScale = [0.5 0.5];
            cf_all{1}.magnSigma2 = 0.00001;
        end
end
gp = gp_set('lik', lik, 'cf', cf_all, 'comp_cf', comp_cf, 'jitterSigma2', jitterSigma, 'latent_method', 'Laplace');

%gradcheck(randn(size(gp_pak(gp))),@gpla_e,@gpla_g,gp,x,y);
%gp = gp_unpak(gp,[gp_pak(gp,'covariance') randn(size(gp_pak(gp,'likelihood')))]);
%gp = gp_unpak(gp,randn(size(gp_pak(gp))));

% k-means clustering initialization
switch init
    case 'default'
        % do nothing but use default initial values
    case 'kmeanInit'
        clust = kmeans(y,gp.lik.RCPnum);
        phi_init=[];
        %figure
        for i1=1:lik.RCPnum
            phi_init(i1,:) = min(max(mean(y(clust==i1,:),1),0.2),0.9);
            %subplot(lik.RCPnum,1,i1),plot(phi_init(i1,:))
        end
        % add some random noise
        phi_init = log(phi_init./(1-phi_init))';
        theta_init = gp_pak(gp,'covariance');
        %gp = gp_unpak(gp, [theta_init phi_init(:)'+0.1*randn(1,numel(phi_init))]);
        gp = gp_unpak(gp, [theta_init phi_init(:)']);
        
        % check that we put the profiles correctly into phi
%         figure
%         phi_init2=reshape(gp.lik.phi, lik.Snum, lik.RCPnum)';
%         %phi_init2=reshape(phi_init(:)', lik.Snum, lik.RCPnum)';
%         for i1=1:lik.RCPnum
%             subplot(lik.RCPnum,1,i1),plot(phi_init2(i1,:))
%             hold on, plot(phi_init(:,i1),'r')
%         end
    case 'totalRand'
        gp = gp_unpak(gp,gp_pak(gp)+0.1*randn(size(gp_pak(gp))));
end

% optimize with SCG
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
gp=gp_optim(gp,x,y,'opt',opt); 
% Optimize with the BFGS quasi-Newton method
%gp=gp_optim(gp,x,y,'opt',opt,'optimf',@fminlbfgs);

% save optimized GP
save(sprintf('optimization_%s%s%s_simulatedData', model, betasigma_fixed, ZeroFirstLatent),'gp','x','y','xpred','model')

% 10-fold cross validation
if docv
    %[criteria, cvpreds, cvws, trpreds, trw, cvtrpreds] = gp_kfcv(gp, x, y, 'k', 10, 'inf_method', 'fixed', 'pred', 'f+lp');
    [criteria, cvpreds, cvws, trpreds, trw, cvtrpreds] = gp_kfcv(gp, x, y, 'k', 10, 'inf_method', 'MAP', 'pred', 'f+lp');
    save(sprintf('simulated_%d', RCPnum))
end
%% Compare CV results: Run this only if you conducted cross validation
if docv
    cv_res = [];
    for i1 = 2:5
        temp = load(sprintf('simulated_%d', i1));
        cv_res(i1,1) = temp.criteria.mlpd_cv;
        cv_res(i1,2) = sqrt(temp.criteria.Var_lpd_cv);
    end
    cv_res
    
    % Difference to true model
    temp = load(sprintf('simulated_%d', 3));
    lpyt_best = temp.cvpreds.lpyt;
    cv_res2 = [];
    for i1 = 2:5
        temp = load(sprintf('simulated_%d', i1));
        diflpyt = lpyt_best - temp.cvpreds.lpyt;
        cv_res2(i1,1) = mean(diflpyt);
        cv_res2(i1,2) = sum(diflpyt>0)/numel(diflpyt);
    end
    cv_res2
end
%%

% Predict into raster
[Ef, Varf] = gpla_pred(gp,x,y,xpred);

% calculate the predictive probabilities for RCPs
S=10000;
[ntest,nout]=size(Ef);
pi=zeros(ntest,nout);
for i1=1:ntest
    Sigm_tmp=(Varf(:,:,i1)'+Varf(:,:,i1))./2;
    f_star=mvnrnd(Ef(i1,:), Sigm_tmp, S);
    
    tmp = exp(f_star);
    tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
    pi(i1,:)=mean(tmp);
    pi05(i1,:)=prctile(tmp,5);
    pi95(i1,:)=prctile(tmp,95);
end

figure,
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesFontSize',8)   %6 8
set(gcf,'DefaultTextFontSize',8)   %6 8)
set(gcf,'DefaultLineLineWidth',1.5)
for i1=1:nout
    f_m = reshape(pi(:,i1),size(X1));
    sp(2+(i1-1)*5) = subplot(3,5,2+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('Pr( RCP = %d )', i1)), set(gca,'YTick',[],'XTick', [])
    f_m = reshape(pi05(:,i1),size(X1));
    sp(1+(i1-1)*5) = subplot(3,5,1+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('RCP %d - lower CI', i1)), set(gca,'YTick',[],'XTick', [])
    f_m = reshape(pi95(:,i1),size(X1));
    sp(3+(i1-1)*5) = subplot(3,5,3+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('RCP %d - upper CI', i1)), set(gca,'YTick',[],'XTick', [])
end
cb(1) = colorbar('southoutside');

% % Plot the classification into RCP areas
% pic = nan(size(pi(:,1)));
% for i1=1:ntest
%     pic(i1) = find(pi(i1,:)==max(pi(i1,:)));
% end
% f_m = reshape(pic,size(X1));
% sp(9) = subplot(3,5,9); pcolor(X1,X2,f_m),shading flat, axis equal
% axis([-0.15 0.15 -0.15 0.15 ])
% title('RCP classification'), set(gca,'YTick',[],'XTick', [])
% colormap(sp(6),parula(nout)); cb(2) = colorbar('southoutside');

% visualize the species probability profiles per RCP
phi = reshape(gp.lik.phi, gp.lik.Snum, gp.lik.RCPnum)';
pS = 1./(1+exp( -phi ));
for i1=1:nout
    sp(5*i1) = subplot(3,5,5*i1);
    plot(pS(i1,:)), grid on
   xlim([0 size(pS,2)])
end
ylabel('Probability of species presence', 'pos', [-14 1.7 -1.0000]);
xlabel('The species identifier')

% Set positions etc.
set(gcf, 'pos', [1 1 20 15])
set(sp(1), 'pos', [0.05 0.66 0.18 0.35])
set(sp(2), 'pos', [0.25 0.66 0.18 0.35])
set(sp(3), 'pos', [0.45 0.66 0.18 0.35])

set(sp(6), 'pos', [0.05 0.35 0.18 0.35])
set(sp(7), 'pos', [0.25 0.35 0.18 0.35])
set(sp(8), 'pos', [0.45 0.35 0.18 0.35])

set(sp(11), 'pos', [0.05 0.05 0.18 0.35])
set(sp(12), 'pos', [0.25 0.05 0.18 0.35])
set(sp(13), 'pos', [0.45 0.05 0.18 0.35])

set(sp(5), 'pos', [0.7 0.71 0.28 0.25], 'xtick',[])
set(sp(10), 'pos', [0.7 0.41 0.28 0.25], 'xtick',[])
set(sp(15), 'pos', [0.7 0.11 0.28 0.25])

set(cb(1),'pos', [0.1 0.07 0.5 0.02])
set(get(cb(1),'xlabel'),'string', 'probability')
% set(cb(2),'pos', [0.7 0.1 0.2 0.03])
% set(get(cb(2),'xlabel'),'string', 'RCP class')
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', sprintf('simulatedData_RCPareas_%s%s%s', model, betasigma_fixed, ZeroFirstLatent)) 
end


%% MCMC analysis

% store the GP of Laplace approximation into its own object
gpla = gp;

% Test sampling options
lik_hmc_opt.steps=4;
lik_hmc_opt.stepadj=0.09;
lik_hmc_opt.nsamples=100;
lik_hmc_opt.persistence=0;
lik_hmc_opt.decay=0.8;
lik_hmc_opt.display=0;

f = gpla_pred(gp,x,y,x);
w = gp_pak(gp,'likelihood');
fe = @(w, lik) (-lik.fh.ll(feval(lik.fh.unpak,lik,w),y,f)-lik.fh.lp(feval(lik.fh.unpak,lik,w)));
fg = @(w, lik) (-lik.fh.llg(feval(lik.fh.unpak,lik,w),y,f,'param')-lik.fh.lpg(feval(lik.fh.unpak,lik,w)));
% Set the state
[w, energies, diagnh] = hmc2(fe, w, lik_hmc_opt, fg, gp.lik);

% Some sampling options
ssls_opt.infer_params = 'covariance';   %  !!! korjaa tämä gp_mc:stä GPstuffiin
ssls_opt.latent_opt.repeat = 20;

latent_opt.repeat = 10;

nsamples = 200;   %  INCREASE THIS if you want more MCMC samples from the full posterior

 % ======= sample only likelihood parameters =================
lik_hmc_opt.nsamples=nsamples;
[w, energies, diagnh] = hmc2(fe, gp_pak(gp,'likelihood'), lik_hmc_opt, fg, gp.lik);

% ======= sample only latent variables and likelihood hyperparameters =================
gp = gp_set(gpla, 'latent_method', 'MCMC');
gp.infer_params = 'likelihood';
lik_hmc_opt.nsamples=50;
[rgpLL,gp,opt]=gp_mc(gp, x, y, 'repeat',5, 'nsamples', nsamples , 'lik_hmc_opt', lik_hmc_opt, 'latent_opt', latent_opt);
%[rgp,gp,opt]=gp_mc(gp, x, y, 'repeat',10, 'nsamples', 5);

% ======= do full MCMC =================
% sampling for both likelihood and covariance function parameters. 
%  Note! This won't work for model = 'polynomial' if betasigma_fixed = '_bsFixed'
%  since in that case there are no covariance function parameters. In this
%  case rgpFULL = rgpLL
if strcmp(model,'polynomial') & strcmp(betasigma_fixed,'_bsFixed')
    rgpFULL = rgpLL
else
    gp = gp_set(gpla, 'latent_method', 'MCMC');
    gp.infer_params = 'covariance+likelihood';
    [rgpFULL,gp,opt]=gp_mc(gp, x, y, 'repeat',5, 'nsamples', nsamples , 'ssls_opt', ssls_opt, 'lik_hmc_opt', lik_hmc_opt, 'latent_opt', latent_opt);
end


rgpLL
rgpFULL
save('simulatedData_mcmc')
%load('simulatedData_mcmc2.mat')
%load('simulatedMCMCruns')


% Predict into raster
%[Ef_mc, Varf_mc] = gp_pred(rgp,x,y,xpred);
Ef_mc = gpmc_preds(rgpFULL,x,y,xpred);

% calculate the predictive probabilities for RCPs
nout = rgpFULL.lik.RCPnum;
ntest=size(Ef_mc,1)./nout;
pi=zeros(ntest,nout);
pi05=zeros(ntest,nout);
pi95=zeros(ntest,nout);
for i1=1:ntest
%     Sigm_tmp=(Varf(:,:,i1)'+Varf(:,:,i1))./2;
%     f_star=mvnrnd(Ef(i1,:), Sigm_tmp, S);
    f_star = Ef_mc(i1:ntest:ntest*(nout-1)+i1,:)';
    tmp = exp(f_star);
    tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
    pi(i1,:)=mean(tmp);
    pi05(i1,:)=prctile(tmp,5);
    pi95(i1,:)=prctile(tmp,95);
end

figure,
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesFontSize',8)   %6 8
set(gcf,'DefaultTextFontSize',8)   %6 8)
set(gcf,'DefaultLineLineWidth',1.5)
for i1=1:nout
    f_m = reshape(pi(:,i1),size(X1));
    sp(2+(i1-1)*5) = subplot(3,5,2+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('Pr( RCP = %d )', i1)), set(gca,'YTick',[],'XTick', [])
    f_m = reshape(pi05(:,i1),size(X1));
    sp(1+(i1-1)*5) = subplot(3,5,1+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('RCP %d - lower CI', i1)), set(gca,'YTick',[],'XTick', [])
    f_m = reshape(pi95(:,i1),size(X1));
    sp(3+(i1-1)*5) = subplot(3,5,3+(i1-1)*5); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    title(sprintf('RCP %d - upper CI', i1)), set(gca,'YTick',[],'XTick', [])
end
cb(1) = colorbar('southoutside');

% % Plot the classification into RCP areas
% pic = nan(size(pi(:,1)));
% for i1=1:ntest
%     pic(i1) = find(pi(i1,:)==max(pi(i1,:)));
% end
% f_m = reshape(pic,size(X1));
% sp(9) = subplot(3,5,9); pcolor(X1,X2,f_m),shading flat, axis equal
% axis([-0.15 0.15 -0.15 0.15 ])
% title('RCP classification'), set(gca,'YTick',[],'XTick', [])
% colormap(sp(6),parula(nout)); cb(2) = colorbar('southoutside');

% visualize the species probability profiles per RCP
phi=[];
pS=[];
for i1 = 1:length(rgpFULL.e)
    phi = reshape(rgpFULL.lik.phi(i1,:), rgpFULL.lik.Snum, rgpFULL.lik.RCPnum)';
    pS(:,:,i1) = 1./(1+exp( -phi ));
end
for i1=1:nout
    sp(5*i1) = subplot(3,5,5*i1); hold on;
    tmp = prctile(squeeze(pS(i1,:,:))', [2.5 50 97.5]);
    for j1 = 1:size(tmp,2)
        plot([j1 j1], [tmp(1,j1) tmp(3,j1)], 'b', 'linewidth', 0.5)
    end
    %plot(1:size(tmp,2),tmp(2,:), 'bo')
    , grid on
   xlim([0 size(pS,2)])
end
ylabel('Probability of species presence', 'pos', [-14 1.7 -1.0000]);
xlabel('The species identifier')

% Set positions etc.
set(gcf, 'pos', [1 1 20 15])
set(sp(1), 'pos', [0.05 0.66 0.18 0.35])
set(sp(2), 'pos', [0.25 0.66 0.18 0.35])
set(sp(3), 'pos', [0.45 0.66 0.18 0.35])

set(sp(6), 'pos', [0.05 0.35 0.18 0.35])
set(sp(7), 'pos', [0.25 0.35 0.18 0.35])
set(sp(8), 'pos', [0.45 0.35 0.18 0.35])

set(sp(11), 'pos', [0.05 0.05 0.18 0.35])
set(sp(12), 'pos', [0.25 0.05 0.18 0.35])
set(sp(13), 'pos', [0.45 0.05 0.18 0.35])

set(sp(5), 'pos', [0.7 0.71 0.28 0.25], 'xtick',[])
set(sp(10), 'pos', [0.7 0.41 0.28 0.25], 'xtick',[])
set(sp(15), 'pos', [0.7 0.11 0.28 0.25])

set(cb(1),'pos', [0.1 0.07 0.5 0.02])
set(get(cb(1),'xlabel'),'string', 'probability')
% set(cb(2),'pos', [0.7 0.1 0.2 0.03])
% set(get(cb(2),'xlabel'),'string', 'RCP class')
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', sprintf('/simulatedData_RCPareas_%s%s_MCMC', model, betasigma_fixed))
end

%% ============= Make comparison figures ==================

%%  ============ 
%  ============ Spatial plots ====================
%  ============ 

% ------------ full MCMC ------------------
% calculate the predictive probabilities for RCPs
nout = rgpFULL.lik.RCPnum;
ntest=size(Ef_mc,1)./nout;
pi=zeros(ntest,nout);
pi05=zeros(ntest,nout);
pi95=zeros(ntest,nout);
for i1=1:ntest
    f_star = Ef_mc(i1:ntest:ntest*(nout-1)+i1,:)';
    tmp = exp(f_star);
    tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
    pi(i1,:)=mean(tmp);
    pi05(i1,:)=prctile(tmp,5);
    pi95(i1,:)=prctile(tmp,95);
end

figure,
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesFontSize',8)   %6 8
set(gcf,'DefaultTextFontSize',8)   %6 8)
set(gcf,'DefaultLineLineWidth',1.5)
sp=[];
for i1=1:nout
    f_m = reshape(pi05(:,i1),size(X1));
    sp(i1,1) = subplot(3,6,1+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title(sprintf('5%% quantile\nMCMC'))
    end
    ylabel(sprintf('RCP %d', i1), 'fontweight', 'bold')
    f_m = reshape(pi(:,i1),size(X1));
    sp(i1,3) = subplot(3,6,3+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title(sprintf('mean\nMCMC'))
    end
    f_m = reshape(pi95(:,i1),size(X1));
    sp(i1,5) = subplot(3,6,5+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title(sprintf('95%% quantile\nMCMC'))
    end
end

% ------- Laplace approximation ----------
% calculate the predictive probabilities for RCPs
S=10000;
[ntest,nout]=size(Ef);
pi=zeros(ntest,nout);
for i1=1:ntest
    Sigm_tmp=(Varf(:,:,i1)'+Varf(:,:,i1))./2;
    f_star=mvnrnd(Ef(i1,:), Sigm_tmp, S);    
    tmp = exp(f_star);
    tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
    pi(i1,:)=mean(tmp);
    pi05(i1,:)=prctile(tmp,5);
    pi95(i1,:)=prctile(tmp,95);
end
for i1=1:nout
    f_m = reshape(pi05(:,i1),size(X1));
    sp(i1,2) = subplot(3,6,2+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title('Laplace')
    end
    f_m = reshape(pi(:,i1),size(X1));
    sp(i1,4) = subplot(3,6,4+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title('Laplace')
    end
    f_m = reshape(pi95(:,i1),size(X1));
    sp(i1,6) = subplot(3,6,6+(i1-1)*6); pcolor(X1,X2,f_m ),shading flat, axis equal
    axis([min(X1(:)) max(X1(:)) min(X2(:)) max(X2(:))]), caxis([0 1])
    set(gca,'YTick',[],'XTick', [])
    if i1==1
        title('Laplace')
    end
end

cb(1) = colorbar('westoutside');
set(gcf, 'pos', [1 1 16 8.5])
set(sp(1,1), 'pos', [0.11 0.58 0.14 0.34])
set(sp(1,2), 'pos', [0.255 0.58 0.14 0.34])
set(sp(1,3), 'pos', [0.41 0.58 0.14 0.34])
set(sp(1,4), 'pos', [0.555 0.58 0.14 0.34])
set(sp(1,5), 'pos', [0.71 0.58 0.14 0.34])
set(sp(1,6), 'pos', [0.855 0.58 0.14 0.34])

set(sp(2,1), 'pos', [0.11 0.30 0.14 0.34])
set(sp(2,2), 'pos', [0.255 0.30 0.14 0.34])
set(sp(2,3), 'pos', [0.41 0.30 0.14 0.34])
set(sp(2,4), 'pos', [0.555 0.30 0.14 0.34])
set(sp(2,5), 'pos', [0.71 0.30 0.14 0.34])
set(sp(2,6), 'pos', [0.855 0.30 0.14 0.34])

set(sp(3,1), 'pos', [0.11 0.02 0.14 0.34])
set(sp(3,2), 'pos', [0.255 0.02 0.14 0.34])
set(sp(3,3), 'pos', [0.41 0.02 0.14 0.34])
set(sp(3,4), 'pos', [0.555 0.02 0.14 0.34])
set(sp(3,5), 'pos', [0.71 0.02 0.14 0.34])
set(sp(3,6), 'pos', [0.855 0.02 0.14 0.34])

set(cb(1),'pos', [0.06 0.15 0.01 0.7])
set(get(cb(1),'xlabel'),'string', 'Categorical parameter, \pi')
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', '-r450', sprintf('simulatedData_%s%s_RCPareas_MCMCLaplaceComparison', model, betasigma_fixed))
end



%%  ============ 
%  ============ latent variable plots ====================
%  ============ 

EfFULL = gpmc_preds(rgpFULL,x,y,xpred);
EfLL = gpmc_preds(rgpLL,x,y,xpred);
[EfLA,VarfLA] = gpla_pred(gpla,x,y,xpred);

tmpFULL = prctile(EfFULL', [5 50 95]);
tmpLL = prctile(EfLL', [5 50 95]);

tmpFULL(2,:) = mean(EfFULL,2)
tmpLL(2,:) = mean(EfLL,2)
%%
figure,
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesFontSize',8)   %6 8
set(gcf,'DefaultTextFontSize',8)   %6 8)
set(gcf,'DefaultLineLineWidth',1.5)

for i1 = 1:3
    sp(i1) = subplot(3,1,i1); hold on;
    plot(1:size(xpred,1),tmpFULL(3,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'b:')
    plot(1:size(xpred,1),tmpFULL(1,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'b:')
    plot(1:size(xpred,1),tmpFULL(2,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'b')
    plot(1:size(xpred,1),tmpLL(3,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'r:')
    plot(1:size(xpred,1),tmpLL(1,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'r:')
    plot(1:size(xpred,1),tmpLL(2,(1:size(xpred,1))+(i1-1)*size(xpred,1)),'r')
    plot(1:size(xpred,1),EfLA(:,i1),'k')    
    plot(1:size(xpred,1),EfLA(:,i1)+1.96*squeeze(sqrt(VarfLA(i1,i1,:))),'k:')
    plot(1:size(xpred,1),EfLA(:,i1)-1.96*squeeze(sqrt(VarfLA(i1,i1,:))),'k:')

    , grid on
   %xlim([0 100])
   ylabel(sprintf('RCP %d\nLatent variable',i1));
   if i1==3
       xlabel('index of spatial location')
   end
end
xlabel('Index of spatial location')

set(gcf, 'pos', [1 1 7.5 8.5])
set(sp(1),'xtick',[], 'pos',[0.17 0.7 0.8 0.29])
set(sp(2),'xtick',[], 'pos',[0.17 0.4 0.8 0.29])
set(sp(3),'pos',[0.17 0.1 0.8 0.29])
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', '-r450', sprintf('simulatedData_%s%s_latentPosterior_MCMCLaplaceComparison', model, betasigma_fixed))
end

set(gcf, 'pos', [1 1 18 8.5])
set(sp(1),'xtick',[], 'pos',[0.07 0.7 0.92 0.29], 'xlim', [0 size(xpred,1)])
set(sp(2),'xtick',[], 'pos',[0.07 0.4 0.92 0.29], 'xlim', [0 size(xpred,1)])
set(sp(3),'pos',[0.07 0.1 0.92 0.29], 'xlim', [0 size(xpred,1)])
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', '-r450', sprintf('simulatedData_%s%s_latentPosterior_MCMCLaplaceComparisonLarge', model, betasigma_fixed))
end


%%  ============ 
%  ============ Species profile plots ====================
%  ============ 

%rgpLL
%rgpFULL


phiFULL=[];
pSFULL=[];
for i1 = 1:length(rgpFULL.e)
    phiFULL(:,:,i1) = reshape(rgpFULL.lik.phi(i1,:), rgpFULL.lik.Snum, rgpFULL.lik.RCPnum)';
    pSFULL(:,:,i1) = 1./(1+exp( -phiFULL(:,:,i1) ));
end
phiLL=[];
pSLL=[];
for i1 = 1:length(rgpLL.e)
    phiLL(:,:,i1) = reshape(rgpLL.lik.phi(i1,:), rgpLL.lik.Snum, rgpLL.lik.RCPnum)';
    pSLL(:,:,i1) = 1./(1+exp( -phiLL(:,:,i1) ));
end
phiL=[];
pSL=[];
for i1 = 1:size(w,1)
    phiL(:,:,i1) = reshape(w(i1,:), rgpLL.lik.Snum, rgpLL.lik.RCPnum)';
    pSL(:,:,i1) = 1./(1+exp( -phiL(:,:,i1) ));
end

phiLA = reshape(gpla.lik.phi, gpla.lik.Snum, gpla.lik.RCPnum)';
pSLA = 1./(1+exp( -phiLA ));



figure,
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesFontSize',8)   %6 8
set(gcf,'DefaultTextFontSize',8)   %6 8)
set(gcf,'DefaultLineLineWidth',1.5)
sp=[];
for i1 = 1:size(pSLA,1)
    sp(i1) = subplot(3,1,i1); hold on;
    tmp = prctile(squeeze(pSFULL(i1,:,:))', [2.5 50 97.5]);
    for j1 = 1:size(tmp,2)
        plot([j1-0.2 j1-0.2], [tmp(1,j1) tmp(3,j1)], 'b', 'linewidth', 0.5)
    end
    plot((1:size(tmp,2))-0.2,tmp(2,:), 'b.')
    tmp = prctile(squeeze(pSLL(i1,:,:))', [2.5 50 97.5]);
    for j1 = 1:size(tmp,2)
        plot([j1+0.2 j1+0.2], [tmp(1,j1) tmp(3,j1)], 'r', 'linewidth', 0.5)
    end
    plot((1:size(tmp,2))+0.2,tmp(2,:), 'r.')
    tmp = prctile(squeeze(pSL(i1,:,:))', [2.5 50 97.5]);
    for j1 = 1:size(tmp,2)
        plot([j1 j1], [tmp(1,j1) tmp(3,j1)], 'k', 'linewidth', 0.5)
    end
    plot(1:size(tmp,2),tmp(2,:), 'k.')
    , grid on
    if i1==2
        ylabel(sprintf('Probability of species presence\n RCP 2'));
    else
        ylabel(sprintf('RCP %d',i1));
    end
    xlim([-1 101])
end
xlabel('The species identifier')

set(gcf, 'pos', [1 1 18 8.5])
set(sp(1),'xtick',[], 'pos',[0.08 0.7 0.91 0.28])
set(sp(2),'xtick',[], 'pos',[0.08 0.4 0.91 0.28])
set(sp(3),'pos',[0.08 0.1 0.91 0.28])
set(gcf, 'paperposition', get(gcf,'pos'))
if printFigure
    print('-dpng', '-r450', sprintf('simulatedData_%s%s_likParamPosterior_MCMCLaplaceComparison', model, betasigma_fixed))
end
