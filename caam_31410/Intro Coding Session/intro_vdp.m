


clear;







% IC (1,0)



dt = 0.005;
Tmax = 100;

t = 0:dt:Tmax;



mu_vec = linspace(0.1,4,100);
T_vec = NaN(size(mu_vec));

for jj=1:length(mu_vec)
    
    mu = mu_vec(jj);
    
    x = NaN(size(t));
    y = NaN(size(t));
    
    x(1) = 1;
    y(1) = 0;
    

    
    
    for ii=2:length(t)
    
        x_prev = x(ii-1);
        y_prev = y(ii-1);
    
        fx = mu*(x_prev-(1/3)*x_prev^3-y_prev);
        fy = (1/mu)*x_prev;
    
        x(ii) = x_prev+dt*fx;
        y(ii) = y_prev+dt*fy;
    
    
    end
    
    
    x(1) = x(end);
    y(1) = y(end);
    
    for ii=2:length(t)
    
        x_prev = x(ii-1);
        y_prev = y(ii-1);
    
        fx = mu*(x_prev-(1/3)*x_prev^3-y_prev);
        fy = (1/mu)*x_prev;
    
        x(ii) = x_prev+dt*fx;
        y(ii) = y_prev+dt*fy;
    
    
    end
    
    
    
    % subplot(1,2,1)
    % plot(t,x)
    % hold on 
    % plot(t,y)
    % 
    % subplot(1,2,2)
    % plot(x,y)
    
    
    
    
    y1 = y(1:end-1);
    y2 = y(2:end);
    
    sign_flips  = t(y1<0 & y2>0);
    
    D = diff(sign_flips);
    
    T = D(end);

    T_vec(jj) = T;

end



plot(mu_vec,T_vec)

