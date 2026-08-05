clc
clear all
close all
fid = fopen('Rx_Calibration_Out.txt','r')
FFT_length=256;
freq=3.3;
lembda=(3e8/(freq*1e9));
xstep=4*49.6*0.001;
ystep=2*50.8*0.001;
finalNfData=zeros(50,8)
while ~feof(fid)
iqdatain=fscanf(fid,'%x');
end
iqdatain1=iqdatain(25:825);
x = iqdatain1;
Ch1IDdata=x(1:16:800);
Ch2IDdata=x(3:16:800);
Ch3IDdata=x(5:16:800);
Ch4IDdata=x(7:16:800);
Ch5IDdata=x(9:16:800);
Ch6IDdata=x(11:16:800);
Ch7IDdata=x(13:16:800);
Ch8IDdata=x(15:16:800);


Ch1QDdata=x(2:16:800);
Ch2QDdata=x(4:16:800);
Ch3QDdata=x(6:16:800);
Ch4QDdata=x(8:16:800);
Ch5QDdata=x(10:16:800);
Ch6QDdata=x(12:16:800);
Ch7QDdata=x(14:16:800);
Ch8QDdata=x(16:16:800);

I1dataUnWrapped=Ch1IDdata;
I1indexG = find(Ch1IDdata>32767);
I1dataUnWrapped(I1indexG) = Ch1IDdata(I1indexG)-(65535+1);
I1=I1dataUnWrapped;

Q1dataUnWrapped=Ch1QDdata;

Q1indexG = find(Ch1QDdata>32767);
Q1dataUnWrapped(Q1indexG) = Ch1QDdata(Q1indexG)-(65535+1);
Q1=Q1dataUnWrapped;

Magnitude1=sqrt((I1.*I1)+(Q1.*Q1));
phase1=atan(Q1./I1)
finalNfData1=Magnitude1.*exp(-1j*phase1)

I2dataUnWrapped=Ch2IDdata;

I2indexG = find(Ch2IDdata>32767);
I2dataUnWrapped(I2indexG) = Ch2IDdata(I2indexG)-(65535+1);
I2=I2dataUnWrapped;

Q2dataUnWrapped=Ch2QDdata;

Q2indexG = find(Ch2QDdata>32767);
Q2dataUnWrapped(Q2indexG) = Ch2QDdata(Q2indexG)-(65535+1);
Q2=Q2dataUnWrapped;

Magnitude2=sqrt((I2.*I2)+(Q2.*Q2));
phase2=atan(Q2./I2)

finalNfData2=Magnitude2.*exp(-1j*phase2)

I3dataUnWrapped=Ch3IDdata;

I3indexG = find(Ch3IDdata>32767);
I3dataUnWrapped(I3indexG) = Ch3IDdata(I3indexG)-(65535+1);
I3=I3dataUnWrapped;

Q3dataUnWrapped=Ch3QDdata;

Q3indexG = find(Ch3QDdata>32767);
Q3dataUnWrapped(Q3indexG) = Ch3QDdata(Q3indexG)-(65535+1);
Q3=Q3dataUnWrapped;

Magnitude3=sqrt((I3.*I3)+(Q3.*Q3));
phase3=atan(Q3./I3)
finalNfData3=Magnitude3.*exp(-1j*phase3)


I4dataUnWrapped=Ch4IDdata;

I4indexG = find(Ch4IDdata>32767);
I4dataUnWrapped(I4indexG) = Ch4IDdata(I4indexG)-(65535+1);
I4=I4dataUnWrapped;

Q4dataUnWrapped=Ch4QDdata;

Q4indexG = find(Ch4QDdata>32767);
Q4dataUnWrapped(Q4indexG) = Ch4QDdata(Q4indexG)-(65535+1);
Q4=Q4dataUnWrapped;

Magnitude4=sqrt((I4.*I4)+(Q4.*Q4));
phase4=atan(Q4./I4)
finalNfData4=Magnitude4.*exp(-1j*phase4)

I5dataUnWrapped=Ch5IDdata;

I5indexG = find(Ch5IDdata>32767);
I5dataUnWrapped(I5indexG) = Ch5IDdata(I5indexG)-(65535+1);
I5=I5dataUnWrapped;

Q5dataUnWrapped=Ch5QDdata;

Q5indexG = find(Ch5QDdata>32767);
Q5dataUnWrapped(Q5indexG) = Ch5QDdata(Q5indexG)-(65535+1);
Q5=Q5dataUnWrapped;

Magnitude5=sqrt((I5.*I5)+(Q5.*Q5));
phase5=atan(Q5./I5)
finalNfData5=Magnitude5.*exp(-1j*phase5)

I6dataUnWrapped=Ch6IDdata;

I6indexG = find(Ch6IDdata>32767);
I6dataUnWrapped(I6indexG) = Ch6IDdata(I6indexG)-(65535+1);
I6=I6dataUnWrapped;

Q6dataUnWrapped=Ch6QDdata;

Q6indexG = find(Ch6QDdata>32767);
Q6dataUnWrapped(Q6indexG) = Ch6QDdata(Q6indexG)-(65535+1);
Q6=Q6dataUnWrapped;

Magnitude6=sqrt((I6.*I6)+(Q6.*Q6));
phase6=atan(Q6./I6)
finalNfData6=Magnitude6.*exp(-1j*phase6)

% m=mean(Magnitude);
% CH6_AMP=(20*log10(m))-(20*log10(32768))-6.3393

I7dataUnWrapped=Ch7IDdata;

I7indexG = find(Ch7IDdata>32767);
I7dataUnWrapped(I7indexG) = Ch7IDdata(I7indexG)-(65535+1);
I7=I7dataUnWrapped

Q7dataUnWrapped=Ch7QDdata;

Q7indexG = find(Ch7QDdata>32767);
Q7dataUnWrapped(Q7indexG) = Ch7QDdata(Q7indexG)-(65535+1);
Q7=Q7dataUnWrapped

Magnitude7=10*log10(sqrt((I7.*I7)+(Q7.*Q7)))
phase7=atan(Q7./I7)
finalNfData7=Magnitude7.*exp(-1j*phase7)
I8dataUnWrapped=Ch8IDdata;


I8indexG = find(Ch8IDdata>32767);
I8dataUnWrapped(I8indexG) = Ch8IDdata(I8indexG)-(65535+1);
I8=I8dataUnWrapped;

Q8dataUnWrapped=Ch8QDdata;

Q8indexG = find(Ch8QDdata>32767);
Q8dataUnWrapped(Q8indexG) = Ch8QDdata(Q8indexG)-(65535+1);
Q8=Q8dataUnWrapped;

Magnitude8=10*log10(sqrt((I8.*I8)+(Q8.*Q8)))
phase8=atan(Q8./I8)
finalNfData8=Magnitude8.*exp(-1j*phase8)
finalNfData=[finalNfData1 finalNfData2 finalNfData3 finalNfData4 finalNfData5 finalNfData6 finalNfData7 finalNfData8]
x_axis_variation=-127:1:128;
y_axis_variation=-127:1:128;
% figure
% pcolor(x_span,y_span,10*log10(power_level)+30)
% title('Power level (in dBm)')
% shading flat
% colorbar
%  
array_factor=fftshift(fft2(finalNfData,FFT_length,FFT_length));  
abs1=abs(array_factor);
abs_transpose=abs1';
figure
pcolor(x_axis_variation,y_axis_variation,20*log10(abs(array_factor)))
title('Far Field Pattrn')
shading flat
colorbar

%%%%%%%%%%%%%---------u v conversion----------%%%%%%%%%%%%%%%%%%
u=([-FFT_length/2:((FFT_length/2)-1)])*lembda/(FFT_length*xstep);
v=([-FFT_length/2:((FFT_length/2)-1)])*lembda/(FFT_length*ystep);

[Real_kx_space]=find(abs(u)<=1 );
[Real_ky_space]=find(abs(v)<=1 );
u=u(Real_kx_space);
v=v(Real_ky_space);
AF=array_factor;
AF1=AF(Real_ky_space,Real_kx_space);
Real_kx=u;
Real_ky=v;
%%%%%%%%%%---------generation of atlas display-------%%%%%%%%%%%%%%%% 
Real_k_space_Patt=[];
            for row=1:length(v);
                for col=1:length(u);
                    if   isreal(sqrt(1-u(col)^2-v(row)^2));
                        Real_k_space_Patt(row,col)=AF1(row,col);
                        
                    else
                        Real_k_space_Patt(row,col)=0;
                       
                    end
                end
            end


AF_on_Atlas= fliplr(atlasDisplay(Real_k_space_Patt,Real_kx,Real_ky));
Pattern_Data(:,:)= AF_on_Atlas(:,:);

Xaxis=asind(Real_kx);
Yaxis=asind(Real_ky);
Az_Points=Xaxis;
El_Points=Yaxis;
Azpoints=length(Xaxis);
Elpoints=length(Yaxis);

[dummy,az]=max(max(abs(AF_on_Atlas)));    % column no. with element having maximum gain
[dummy,el]=max(max(abs((AF_on_Atlas)'))); % row no. with element having maximum gain
MAX=(max(max(abs(AF_on_Atlas))));
divide=(abs(AF_on_Atlas))/MAX;

pcolor(Xaxis,Yaxis,20*log10(abs(AF_on_Atlas/(max(max(AF_on_Atlas))))))
shading flat
colorbar
xlabel('Theta (Degrees)->','FontSize',14);
ylabel('Phi (Degrees)->','FontSize',14);
title('Intensity Az El ','FontSize',14)
%%%%%%%%-------Azimuth Cut----------%%%%%%%%%%%%%%%%
f5 = figure('NumberTitle','off','Name','Azimuth Probe Compensation Comparision');
plot(Az_Points,20*log10(abs(AF_on_Atlas(el,:))/MAX),'b');
hold on
grid on
title('Azimuth Cut','FontSize',14)
xlim([-90 90]) ;
ylim([-30 0]) ;

% %%%%%%------ElevationProbe Compensation Comparision-------%%%%%%%%%
f6 = figure('NumberTitle','off','Name','ElevationProbe Compensation Comparision');
plot(El_Points,20*log10(abs(AF_on_Atlas(:,az))/MAX),'r');
hold on
grid on
title('Elevation Cut ','FontSize',14)
xlim([-90 90]) ;
ylim([-50 0]) ;
% m=mean(Magnitude)
% CH7_AMP=(20*log10(m))-(20*log10(32768))-6.3393
% fclose(fid);
% hold on ;
% figure(1);
% % title( 'CH7 Q DATA');
%  AXIS([0 50 -65 65])
%  grid on;
%   XLABEL('SAMPLES');
%   YLABEL('AMPLITUDE');
% plot(Q7dataUnWrapped);
% figure(2);
% title( 'CH7 I DATA');
%  AXIS([0 50 -65 65])
%  grid on;
%   XLABEL('SAMPLES');
%   YLABEL('AMPLITUDE');
% hold on;
% plot(I7dataUnWrapped);

