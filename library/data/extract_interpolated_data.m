%% Extract data
load('/work1/s174505/Thesis/Data/raw_data/20150708_162643_tgipar.mat');
load('/work1/s174505/Thesis/Data/raw_data/20150708_163106_dypar.mat');

curv = datigeo(:, 6); % [1/m]
scartamento = datigeo(:, 4) * 1e-3; % [m]
y = datigeo(:, [15:18, 27:34]);
Xold = fillmissing(dati(:,[2 4:39 68]), 'linear');

%% Fix known data issues
fixed_pos = ((0:size(datigeo,1)-1)/2)'; % [m]
datigeo(:,1) = fixed_pos;
dati(~isnan(dati(:,1)),1) = fixed_pos;

Xold(775066:775149, 3) = 0;
Xold(1266670:1266750, 7) = 0;
Xold(:, end) = nan;
Xold(~isnan(dati(:,1)), end) = curv;
Xold(:, end) = fillmissing(Xold(:, end), 'linear');

old_pos = fillmissing(dati(:,1),'linear');
new_pos = ((2:1:(6*fixed_pos(end)-2))/6)';
Xnew = zeros(length(new_pos),size(Xold,2));
%% Interpolate data and crop y

for i = 1:size(Xold, 2)
    Vq = griddedInterpolant(old_pos,Xold(:,i),'spline');
    Xnew(:,i) = Vq(new_pos);
end

position = fixed_pos(2:end-1);
y = y(2:end-1, :);

%% Split data into training and testing
X_train  = [Xnew((3*23941+1):(3*93941), :); Xnew((3*117881+1):(3*187881), :); Xnew((3*211821+1):(3*281821), :)];
X_test   = [Xnew((3*1+1):(3*23941), :); Xnew((3*93941+1):(3*117881), :); Xnew((3*187881+1):(3*211821), :)];
X_train_v2 = permute(reshape(X_train.', 38, 210000, []),[3,1,2]);
X_train  = permute(reshape(X_train.', 38, 3000, []),[3,1,2]);
X_test   = permute(reshape(X_test.', 38, 71820, []),[3,1,2]);

y_train  = [y(23942:93941, :); y(117882:187881, :); y(211822:281821, :)];
y_test   = [y(2:23941, :); y(93942:117881, :); y(187882:211821, :)];
y_train_v2 = permute(reshape(y_train.', 12, 70000, []), [3,1,2]);
y_train = permute(reshape(y_train.', 12, 1000, []), [3,1,2]);
y_test   = permute(reshape(y_test.', 12, 23940, []), [3,1,2]);

%% Save data
save('/work1/s174505/Thesis/Data/Interpolated_data.mat', 'X_train', 'y_train', 'X_test', 'y_test');
X_train = X_train_v2;
y_train = y_train_v2;
save('/work1/s174505/Thesis/Data/Interpolated_data_v2.mat', 'X_train', 'y_train', 'X_test', 'y_test');