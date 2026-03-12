%% Numerical Experiment - MODTRAN Atmosphere Version
% Based on numerical_experiment_matlab_simple.m
% Uses MODTRAN atmospheric data for realistic spectral irradiance
% instead of the simplified 70/30 direct/diffuse Gaussian model.

clear; clc;

%% Setup
scope_dir = fileparts(mfilename('fullpath'));
cd(scope_dir);
addpath(genpath(fullfile(scope_dir, 'src')));

%% Define parameter values (Table 2)
Cab_values = [10, 20, 40, 80];
LAI_values = [0.5, 1, 3, 6];
LAD_types = {'spherical', 'planophile', 'erectophile'};
tts_values = [30, 45, 60];
tto_values = [0, 20, 40, 60];
Rin_values = [100, 300, 500, 800];
Ta_values = [15, 25, 35];
Vcmax_values = [30, 100, 160];
soil_types = {'zero', 'wet', 'dry_bright1', 'dry_bright2'};

%% Calculate total
n_total = length(Cab_values) * length(LAI_values) * length(LAD_types) * ...
          length(tts_values) * length(tto_values) * length(Rin_values) * ...
          length(Ta_values) * length(Vcmax_values) * length(soil_types);
fprintf('Total scenarios: %d\n', n_total);

%% Initialize spectral and constants
[spectral] = define_bands();
constants = define_constants();

% Load optipar for Fluspect
% Use same file as Python for fair comparison
load(fullfile(scope_dir, 'input', 'fluspect_parameters', 'Optipar2021_ProspectPRO_CX.mat'));

% Fixed canopy parameters
canopy_fixed.nlincl = 13;
canopy_fixed.nlazi = 36;
canopy_fixed.litab = [5:10:75, 81:2:89]';
canopy_fixed.lazitab = (5:10:355);

%% Load MODTRAN atmospheric data ONCE
% This is the key difference from numerical_experiment_matlab_simple.m
% The MODTRAN file provides realistic atmospheric transmission data
atmfile = fullfile(scope_dir, 'input', 'radiationdata', 'FLEX-S3_std.atm');
fprintf('Loading MODTRAN atmospheric data from: %s\n', atmfile);
atmo = load_atmo(atmfile, spectral.SCOPEspec);
fprintf('Loaded MODTRAN data with M matrix size: %d x %d\n', size(atmo.M, 1), size(atmo.M, 2));

%% Preallocate results
results = cell(n_total, 30);
headers = {'Cab', 'LAI', 'LAD', 'tts', 'tto', 'Rin', 'Ta', 'Vcmax', ...
           'soil_type', 'F684', 'F685', 'F740', 'F761', 'wl685', 'wl740', ...
           'LoutF', 'EoutF', 'Eouto', ...
           'Rntot', 'lEtot', 'Htot', 'Actot', 'Rnctot', 'Tcave', 'Tsave', ...
           'escape_685', 'escape_740', 'escape_761', 'group', 'error'};

%% Preallocate spectral outputs
% Store wavelengths and spectral arrays
nwl = length(spectral.wlS);
nwlF = length(spectral.wlF);  % Fluorescence wavelengths
spectral_data.wl = spectral.wlS;           % Full wavelength array [nwl]
spectral_data.wlF = spectral.wlF;          % Fluorescence wavelengths [nwlF]
spectral_data.refl = NaN(n_total, nwl);    % Reflectance spectrum [n_total x nwl]
spectral_data.Lo_ = NaN(n_total, nwl);     % Outgoing radiance [n_total x nwl]
spectral_data.Eout_ = NaN(n_total, nwl);   % Upwelling radiation [n_total x nwl]
spectral_data.LoF_ = NaN(n_total, nwlF);   % Fluorescence radiance [n_total x nwlF]
spectral_data.EoutF_ = NaN(n_total, nwlF); % Hemispherical fluorescence [n_total x nwlF]
spectral_data.LoF_sunlit = NaN(n_total, nwlF);    % Sunlit leaf fluorescence [n_total x nwlF]
spectral_data.LoF_shaded = NaN(n_total, nwlF);    % Shaded leaf fluorescence [n_total x nwlF]
spectral_data.LoF_scattered = NaN(n_total, nwlF);  % Scattered fluorescence [n_total x nwlF]
spectral_data.LoF_soil = NaN(n_total, nwlF);       % Soil-reflected fluorescence [n_total x nwlF]
spectral_data.Femleaves_ = NaN(n_total, nwlF);     % Total leaf fluorescence emission [n_total x nwlF]
spectral_data.escape_ratio_ = NaN(n_total, nwlF);  % Escape ratio EoutF_/Femleaves_ [n_total x nwlF]
spectral_data.Esun_ = NaN(n_total, nwl);   % Direct solar [n_total x nwl]
spectral_data.Esky_ = NaN(n_total, nwl);   % Diffuse sky [n_total x nwl]

%% Run experiment
idx = 0;
start_time = tic;
errors = 0;

for soil_idx = 1:length(soil_types)
    soil_type = soil_types{soil_idx};
    soil_refl = create_soil_spectrum(soil_type, length(spectral.wlS));
    group = (soil_idx > 1) + 1;

    fprintf('\n=== Soil: %s (group %d) ===\n', soil_type, group);

    for i1 = 1:length(Cab_values)
        Cab = Cab_values(i1);
        for i2 = 1:length(LAI_values)
            LAI = LAI_values(i2);
            for i3 = 1:length(LAD_types)
                LAD = LAD_types{i3};
                for i4 = 1:length(tts_values)
                    tts = tts_values(i4);
                    for i5 = 1:length(tto_values)
                        tto = tto_values(i5);
                        for i6 = 1:length(Rin_values)
                            Rin = Rin_values(i6);
                            for i7 = 1:length(Ta_values)
                                Ta = Ta_values(i7);
                                for i8 = 1:length(Vcmax_values)
                                    Vcmax = Vcmax_values(i8);

                                    idx = idx + 1;

                                    try
                                        % Run with MODTRAN atmosphere
                                        [out, spec] = run_single_scenario_mod(Cab, LAI, LAD, tts, tto, ...
                                            Rin, Ta, Vcmax, soil_refl, spectral, constants, ...
                                            optipar, canopy_fixed, atmo);

                                        results(idx, :) = {Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax, ...
                                            soil_type, out.F684, out.F685, out.F740, out.F761, ...
                                            out.wl685, out.wl740, out.LoutF, out.EoutF, out.Eouto, ...
                                            out.Rntot, out.lEtot, out.Htot, out.Actot, ...
                                            out.Rnctot, out.Tcave, out.Tsave, ...
                                            out.escape_685, out.escape_740, out.escape_761, group, ''};

                                        % Store spectral outputs
                                        spectral_data.refl(idx, :) = spec.refl(:)';
                                        spectral_data.Lo_(idx, :) = spec.Lo_(:)';
                                        spectral_data.Eout_(idx, :) = spec.Eout_(:)';
                                        spectral_data.Esun_(idx, :) = spec.Esun_(:)';
                                        spectral_data.Esky_(idx, :) = spec.Esky_(:)';
                                        if ~isempty(spec.LoF_)
                                            spectral_data.LoF_(idx, :) = spec.LoF_(:)';
                                            spectral_data.EoutF_(idx, :) = spec.EoutF_(:)';
                                            spectral_data.LoF_sunlit(idx, :) = spec.LoF_sunlit(:)';
                                            spectral_data.LoF_shaded(idx, :) = spec.LoF_shaded(:)';
                                            spectral_data.LoF_scattered(idx, :) = spec.LoF_scattered(:)';
                                            spectral_data.LoF_soil(idx, :) = spec.LoF_soil(:)';
                                            spectral_data.Femleaves_(idx, :) = spec.Femleaves_(:)';
                                            if ~isempty(spec.escape_ratio_)
                                                spectral_data.escape_ratio_(idx, :) = spec.escape_ratio_(:)';
                                            end
                                        end

                                    catch ME
                                        errors = errors + 1;
                                        if errors <= 3
                                            fprintf('\n=== ERROR %d ===\n', errors);
                                            fprintf('Message: %s\n', ME.message);
                                            if ~isempty(ME.stack)
                                                fprintf('Function: %s, Line: %d\n', ME.stack(1).name, ME.stack(1).line);
                                            end
                                            fprintf('===================\n\n');
                                        end
                                        results(idx, :) = {Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax, ...
                                            soil_type, NaN, NaN, NaN, NaN, NaN, NaN, ...
                                            NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
                                            NaN, NaN, NaN, group, ME.message};
                                    end

                                    if mod(idx, 2000) == 0
                                        elapsed = toc(start_time);
                                        fprintf('  %d/%d (%.1f%%) - %.1f/s - ETA: %.1f min\n', ...
                                            idx, n_total, 100*idx/n_total, idx/elapsed, (n_total-idx)/(idx/elapsed)/60);
                                    end

                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Save results
elapsed = toc(start_time);
fprintf('\n=== DONE ===\n');
fprintf('Total: %d, Errors: %d, Time: %.1f min\n', idx, errors, elapsed/60);

% Save scalar results to CSV
T = cell2table(results, 'VariableNames', headers);
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
outfile = fullfile(scope_dir, 'output', sprintf('numerical_experiment_matlab_modtran_%s.csv', timestamp));
writetable(T, outfile);
fprintf('Saved scalar results: %s\n', outfile);

% Save spectral data to MAT file
specfile = fullfile(scope_dir, 'output', sprintf('numerical_experiment_matlab_modtran_spectral_%s.mat', timestamp));
save(specfile, 'spectral_data', 'headers', '-v7.3');
fprintf('Saved spectral data: %s\n', specfile);
fprintf('  Spectral arrays size: %d scenarios x %d wavelengths\n', n_total, length(spectral_data.wl));
fprintf('  Fluorescence arrays size: %d scenarios x %d wavelengths\n', n_total, length(spectral_data.wlF));


%% ========== Helper Functions ==========

function [LIDFa, LIDFb] = get_lidf_params(lad_type)
    switch lad_type
        case 'spherical'
            LIDFa = -0.35; LIDFb = -0.15;
        case 'planophile'
            LIDFa = 1.0; LIDFb = 0.0;
        case 'erectophile'
            LIDFa = -1.0; LIDFb = 0.0;
        otherwise
            error('Unknown LAD type: %s', lad_type);
    end
end


function soil_refl = create_soil_spectrum(soil_type, nwl)
    switch soil_type
        case 'zero'
            soil_refl = zeros(nwl, 1);
        case 'wet'
            soil_refl = 0.05 * ones(nwl, 1);
        case 'dry_bright1'
            soil_refl = 0.15 * ones(nwl, 1);
        case 'dry_bright2'
            soil_refl = 0.25 * ones(nwl, 1);
        otherwise
            error('Unknown soil type: %s', soil_type);
    end
end


function [out, spec] = run_single_scenario_mod(Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax, ...
    soil_refl, spectral, constants, optipar, canopy_fixed, atmo)
    % Run SCOPE for a single scenario using MODTRAN atmosphere
    % This version uses pre-loaded MODTRAN data (atmo.M) instead of
    % the simplified Gaussian atmosphere model.
    %
    % The MODTRAN matrix atmo.M contains atmospheric transmission data
    % that RTMo uses to compute Esun_ and Esky_, scaling by meteo.Rin
    %
    % Returns:
    %   out  - scalar outputs (fluxes, temperatures, etc.)
    %   spec - spectral outputs (refl, Lo_, Eout_, LoF_, Esun_, Esky_)

    [LIDFa, LIDFb] = get_lidf_params(LAD);

    %% leafbio structure
    leafbio.Cab = Cab;
    leafbio.Cca = Cab / 4;
    leafbio.Cdm = 0.012;
    leafbio.Cw = 0.009;
    leafbio.Cs = 0.0;
    leafbio.Cant = 0.0;
    leafbio.Cbc = 0.0;
    leafbio.Cp = 0.0;
    leafbio.N = 1.4;
    leafbio.fqe(1) = 0.01;
    leafbio.fqe(2) = 0.01;
    leafbio.V2Z = 0;
    leafbio.Vcmax25 = Vcmax;
    leafbio.BallBerrySlope = 9.0;
    leafbio.BallBerry0 = 0.01;
    leafbio.Type = 'C3';
    leafbio.RdPerVcmax25 = 0.015;
    leafbio.Kn0 = 2.48;
    leafbio.Knalpha = 2.83;
    leafbio.Knbeta = 0.114;
    leafbio.Tyear = 15;
    leafbio.beta = 0.51;
    leafbio.kNPQs = 0;
    leafbio.qLs = 0;
    leafbio.stressfactor = 1;
    leafbio.rho_thermal = 0.01;
    leafbio.tau_thermal = 0.01;
    leafbio.emis = 1 - leafbio.rho_thermal - leafbio.tau_thermal;
    leafbio.TDP = define_temp_response_biochem();

    %% canopy structure
    canopy.LAI = LAI;
    canopy.hc = 2.0;
    canopy.LIDFa = LIDFa;
    canopy.LIDFb = LIDFb;
    canopy.leafwidth = 0.1;
    canopy.rb = 10;
    canopy.Cd = 0.3;
    canopy.CR = 0.35;
    canopy.CD1 = 20.6;
    canopy.Psicor = 0.2;
    canopy.rwc = 1;
    canopy.kV = 0.6396;
    canopy.hot = canopy.leafwidth / canopy.hc;
    canopy.zo = 0.25 * canopy.hc;
    canopy.d = 0.65 * canopy.hc;
    canopy.nlayers = max(2, ceil(10 * LAI));
    nl = canopy.nlayers;
    x = (-1/nl : -1/nl : -1)';
    canopy.xl = [0; x];
    canopy.nlincl = canopy_fixed.nlincl;
    canopy.nlazi = canopy_fixed.nlazi;
    canopy.litab = canopy_fixed.litab;
    canopy.lazitab = canopy_fixed.lazitab;
    canopy.lidf = leafangles(canopy.LIDFa, canopy.LIDFb);

    %% soil structure
    soil.spectrum = 1;
    soil.rss = 500;
    soil.rs_thermal = 0.06;
    soil.cs = 1180;
    soil.rhos = 1800;
    soil.CSSOIL = 0.01;
    soil.lambdas = 1.55;
    soil.SMC = 0.25;
    soil.rbs = 10;
    soil.BSMBrightness = 0.5;
    soil.BSMlat = 25;
    soil.BSMlon = 45;
    soil.GAM = Soil_Inertia0(soil.cs, soil.rhos, soil.lambdas);
    soil.Tsold = Ta * ones(12, 2);
    soil.refl = soil_refl;
    soil.refl(spectral.IwlT) = soil.rs_thermal;

    %% meteo structure
    % Rin is used by RTMo to scale the MODTRAN spectral shape
    meteo.Rin = Rin;
    meteo.Rli = 300;
    meteo.Ta = Ta;
    meteo.p = 970;
    meteo.ea = 15;
    meteo.u = 2;
    meteo.Ca = 410;
    meteo.Oa = 209;
    meteo.z = 10;

    %% angles structure
    angles.tts = tts;
    angles.tto = tto;
    angles.psi = 0;

    %% options structure
    options.calc_fluor = 1;
    options.calc_planck = 0;
    options.calc_xanthophyllabs = 0;
    options.soilspectrum = 0;
    options.Fluorescence_model = 0;
    options.calc_directional = 0;
    options.calc_vert_profiles = 0;
    options.calc_ebal = 1;
    options.lite = 1;
    options.verify = 0;
    options.saveCSV = 0;
    options.simulation = 0;
    options.apply_T_corr = 1;
    options.MoninObukhov = 0;
    options.soil_heat_method = 2;
    options.calc_rss_rbs = 0;

    integr = 'layers';

    %% xyt structure
    xyt.t = 0;
    xyt.year = 2024;

    %% mSCOPE layers
    mly.nly = 1;
    mly.pLAI = canopy.LAI;
    mly.totLAI = canopy.LAI;
    mly.pCab = leafbio.Cab;
    mly.pCca = leafbio.Cca;
    mly.pCdm = leafbio.Cdm;
    mly.pCw = leafbio.Cw;
    mly.pCs = leafbio.Cs;
    mly.pN = leafbio.N;

    %% Run Fluspect
    leafopt = fluspect_mSCOPE(mly, spectral, leafbio, optipar, nl);
    leafopt.refl(:, spectral.IwlT) = leafbio.rho_thermal;
    leafopt.tran(:, spectral.IwlT) = leafbio.tau_thermal;

    %% RTMo with MODTRAN atmosphere
    % Pass the pre-loaded MODTRAN atmo structure directly
    % RTMo will use atmo.M to compute Esun_ and Esky_ with realistic
    % atmospheric absorption features, scaling by meteo.Rin
    [rad, gap, profiles] = RTMo(spectral, atmo, soil, leafopt, canopy, angles, constants, meteo, options);

    %% Energy balance
    k = 1;
    [iter, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo] = ...
        ebal(constants, options, rad, gap, meteo, soil, canopy, leafbio, k, xyt, integr);

    %% RTMf
    if options.calc_fluor && isfield(leafopt, 'Mb') && ~isempty(leafopt.Mb)
        [rad] = RTMf(constants, spectral, rad, soil, leafopt, canopy, gap, angles, bcu.eta, bch.eta);
        out.F684 = rad.F684;
        out.F685 = rad.F685;
        out.F740 = rad.F740;
        out.F761 = rad.F761;
        out.wl685 = rad.wl685;
        out.wl740 = rad.wl740;
        out.LoutF = rad.LoutF;
        out.EoutF = rad.EoutF;
        spec.LoF_ = rad.LoF_;
        spec.EoutF_ = rad.EoutF_;
        spec.LoF_sunlit = rad.LoF_sunlit;
        spec.LoF_shaded = rad.LoF_shaded;
        spec.LoF_scattered = rad.LoF_scattered;
        spec.LoF_soil = rad.LoF_soil;
        spec.Femleaves_ = rad.Femleaves_;
        % Escape ratio = EoutF_ / Femleaves_ (per wavelength)
        escape_ratio_ = NaN(size(spec.Femleaves_));
        valid = spec.Femleaves_ > 0;
        escape_ratio_(valid) = spec.EoutF_(valid) ./ spec.Femleaves_(valid);
        spec.escape_ratio_ = escape_ratio_;
        % wlF starts at 640, 1nm steps (MATLAB 1-based): 685nm=46, 740nm=101, 761nm=122
        out.escape_685 = escape_ratio_(46);
        out.escape_740 = escape_ratio_(101);
        out.escape_761 = escape_ratio_(122);
    else
        out.F684 = NaN;
        out.F685 = NaN;
        out.F740 = NaN;
        out.F761 = NaN;
        out.wl685 = NaN;
        out.wl740 = NaN;
        out.LoutF = NaN;
        out.EoutF = NaN;
        out.escape_685 = NaN;
        out.escape_740 = NaN;
        out.escape_761 = NaN;
        spec.LoF_ = [];
        spec.EoutF_ = [];
        spec.LoF_sunlit = [];
        spec.LoF_shaded = [];
        spec.LoF_scattered = [];
        spec.LoF_soil = [];
        spec.Femleaves_ = [];
        spec.escape_ratio_ = [];
    end

    %% Scalar Outputs
    out.Eouto = rad.Eouto;
    out.Rntot = fluxes.Rntot;
    out.lEtot = fluxes.lEtot;
    out.Htot = fluxes.Htot;
    out.Actot = fluxes.Actot;
    out.Rnctot = fluxes.Rnctot;
    out.Tcave = fluxes.Tcave;
    out.Tsave = fluxes.Tsave;

    %% Spectral Outputs
    spec.refl = rad.refl;      % Reflectance spectrum [nwl]
    spec.Lo_ = rad.Lo_;        % Outgoing radiance [nwl]
    spec.Eout_ = rad.Eout_;    % Upwelling radiation [nwl]
    spec.Esun_ = rad.Esun_;    % Direct solar irradiance [nwl]
    spec.Esky_ = rad.Esky_;    % Diffuse sky irradiance [nwl]
end
