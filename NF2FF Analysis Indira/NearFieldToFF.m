clc
clear all
close all
opengl('software')
fid = fopen('amplitudeMPR.txt','r')
FFT_length=512;
freq=3300;
lembda=(3e8/(freq*1e6));
xstep=0.044;
ystep=0.044;
% finalNfData=zeros(50,8)
while ~feof(fid)
amp=fscanf(fid,'%f');
end
amp=amp(1:length(amp));

 amp1=zeros(92,115);
 amp1=reshape(amp,92,115);

fid1 = fopen('phaseMPR.txt','r')  

while ~feof(fid1)
phase1=fscanf(fid1,'%f');
end
phase2=phase1(1:length(phase1));
phase3=exp(-1j*phase2*(pi/180));
phase4=reshape(phase3,92,115);
finalNfData=amp1.*phase4;
finalNfData=fliplr(finalNfData);

x_axis_variation=-255:1:256;
y_axis_variation=-255:1:256;
% figure
% pcolor(x_span,y_span,10*log10(power_level)+30)
% title('Power level (in dBm)')
% shading flat
% colorbar
 
array_factor=fftshift(fft2(finalNfData,FFT_length,FFT_length));  
abs1=abs(array_factor);
abs_transpose=abs1';
% figure
% pcolor(x_axis_variation,y_axis_variation,20*log10(abs(array_factor)))
% title('Far Field Pattrn')
% shading flat
% colorbar

%%%%%%%%%%%%---------u v conversion----------%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%---------generation of atlas display-------%%%%%%%%%%%%%%%% 
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

[~,az]=max(max(abs(AF_on_Atlas)));    % column no. with element having maximum gain
[dummy,el]=max(max(abs((AF_on_Atlas)'))); % row no. with element having maximum gain
MAX=(max(max(abs(AF_on_Atlas))));
divide=(abs(AF_on_Atlas))/MAX;

pcolor(Xaxis,Yaxis,20*log10(abs(AF_on_Atlas/(max(max(AF_on_Atlas))))))
shading flat
colorbar

xlabel('Theta (Degrees)->','FontSize',14);
ylabel('Phi (Degrees)->','FontSize',14);
title('Intensity Az El ','FontSize',14)
%%%%%%%-------Azimuth Cut----------%%%%%%%%%%%%%%%%
theta2=-90:0.01:90;
dbAz=(20*log10(abs(AF_on_Atlas(el,:))/MAX));
dbAz=interp1(Az_Points,dbAz, theta2);
f5 = figure('NumberTitle','off','Name','Azimuth Cut');
plot(theta2,dbAz,'b');

dB_max=max(find(dbAz>=-3));
dB_min=min(find(dbAz>=-3));
Az3dB_bw=theta2(:,dB_max)-theta2(:,dB_min)

hold on
grid on
title('Azimuth Cut','FontSize',14)
xlim([-90 90]) ;
ylim([-70 0]) ;

%%%%%%------ElevationProbe Compensation Comparision-------%%%%%%%%%
f6 = figure('NumberTitle','off','Name','Elevation Cut');
plot(El_Points,20*log10(abs(AF_on_Atlas(:,az))/MAX),'r');
hold on
grid on
title('Elevation Cut ','FontSize',14)
xlim([-90 90]) ;
ylim([-70 0]) ;
