

clear;



dt = 0.1;
Tmax = 10;


t = 0:dt:Tmax;


x = NaN(size(t));


x(1) = 1;




for ii=2:length(t)


    x_prev = x(ii-1);

    f = -x_prev;

    x(ii) = x_prev+dt*f;



end



plot(t,x)

hold on

t_fine = linspace(0,Tmax,10000);
plot(t_fine,exp(-t_fine),"r--")
