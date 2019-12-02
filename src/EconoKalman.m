clear;
cpi_data = load('../data/allitems.CSV',';');
cpi_data = reshape(cpi_data',1,size(cpi_data,1)*size(cpi_data,2));
cpi_data = cpi_data';
inflation_data = zeros(length(cpi_data),1);
for i = 1:1:length(cpi_data)
    if (i ~= 1)
        inflation_data(i) = (cpi_data(i)-cpi_data(i-1))/cpi_data(i-1);
    else
        inflation_data(i) = (cpi_data(i) - 100)/100;
    end
end


p = 2;


train_test_split_ratio = 0.1;
split = floor(length(inflation_data)*train_test_split_ratio);


train_data = inflation_data(1:split);
test_data = inflation_data(split+1:end);
[a,err] = covm(train_data,p);
initial_estimate = mean(train_data)*ones(p,1);
var = cov(train_data);
[r, lags] = xcorr(inflation_data,inflation_data);
r = r(length(inflation_data):end);
P = zeros(p,p); % Error Covariance Matrix
for i = 1:1:p
    for k = 1:1:p
        P(i,k) = r(abs(k-i)+1);
    end
end

stationary_window_length = 30;
A = [(a(2:end))'; horzcat(eye(p-1),zeros(p-1,1))];
C = ones(1,p);
x_record = zeros(length(train_data),1);
x_record_predict = zeros(length(train_data),1);
samples = zeros(stationary_window_length,1);
for n = 1:1:length(test_data)-p+1
    if n == 1
        x = initial_estimate;   
    else 
        x_record(n) = x(1);
        x = A*x;
        x_record_predict(n) = x(1);
        P = A*P*A' + eye(p)*err;
        K = P*C'*inv(C*P*C');
        y = C*test_data(n:n+p-1);
        if rem(n,stationary_window_length) == 0
            [a,err] = covm(samples,p);
            A = [(a(2:end))'; horzcat(eye(p-1),zeros(p-1,1))];
        else
            samples(rem(n,stationary_window_length)) = y(1);
        end
        x = x + K*(y - C*x);
        P = (eye(p) - K*C)*P;
    end
end

hold on;
plot(test_data);
plot(-x_record_predict);
plot(x_record);
legend(["Real data" "Initial Prediction" "Final Precition"])
%legend(["Real data" "Filtered"])
corrcoef(x_record(stationary_window_length:end),test_data(stationary_window_length:end-p+1))
corrcoef(x_record_predict(stationary_window_length:end),test_data(stationary_window_length:end-p+1))