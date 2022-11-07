%% Load data
load('20150708_162643_tgipar.mat');
load('20150708_163106_dypar.mat');

dati_cleaned = dati(~isnan(dati(:,1)),[2 4:68]);
datigeo_cleaned = datigeo(:,[2 4:34]);

%% log sum abs
disp('Log of sum of abs');
disp('dati_cleaned');
disp(log(sum(abs(dati_cleaned))));
disp('datigeo');
disp(log(sum(abs(datigeo_cleaned))));

%% Means
disp('Means');
disp('dati_cleaned');
disp(mean(dati_cleaned));
disp('datigeo');
disp(mean(datigeo_cleaned));

%% Std
disp('Std');
disp('dati_cleaned');
disp(std(dati_cleaned));
disp('datigeo');
disp(std(datigeo_cleaned));

%% Var
disp('Var');
disp('dati_cleaned');
disp(var(dati_cleaned));
disp('datigeo');
disp(var(datigeo_cleaned));

%% Corr
disp('Corr');
disp('dati_cleaned');
disp(corr(dati_cleaned));
disp('datigeo');
disp(corr(datigeo_cleaned));

