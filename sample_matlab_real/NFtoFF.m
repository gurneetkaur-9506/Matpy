 %Near Field Simulation to forsee disturbances in the antenna aperture done
%by Indira Srivastava and Ankit Thalor under the guidance of U.S. Pandey
%Sc.E
clc
close all;
clear all;
%%%%%%%%%------- put excitation=0 for uniform excitation-----%%%%%%%%%
%%%%%%%%%------- distance of probe from antenna 455mm---------%%%%%%%%
%excitation=0;   
%excitation1=10;
%d=455;              
freq=1.03;
%%%%%%%%-------- wavelength -------------%%%%%%%%%%%%%%%%%%%%%%%%
lembda=(3e8/(freq*1e6)); 
FFT_length=512;

%%%%%%%--------probe plane ---------%%%%%%%%%%%%%%
scan_start_x = -2000;
scan_start_y = -1000; 
scan_stop_x = 2000;
scan_stop_y = 1000;

%%%%%%--------antenna transmit plane --------%%%%%%%%%%%%%
trans_start_x = -600;
trans_start_y = -400;
trans_stop_x = 600;
trans_stop_y = 400;

%%%%%%----probe steps ---------%%%%%%%%%%%%%%%%%s
xstart=0;
xstep=44;
xstop=(scan_stop_x)-(scan_start_x);
x_span= xstart:xstep:xstop;
length_x=length(x_span);

ystart=0;
ystep=44;
ystop=(scan_stop_y)-(scan_start_y);
y_span= ystart:ystep:ystop;
length_y=length(y_span);

%%%%%%%%%%----antenna aperture---------%%%%%%%%%%%%%%
%delta_x=24.8;
%delta_y=50.8;
%m=nearest((trans_stop_y-trans_start_y ) /delta_y);
%n=nearest((trans_stop_x-trans_start_x ) /delta_x);


%element_power=zeros(m,n);
%element_excitation=zeros(m,n);
%%%%%%%%%%------phase calculation------%%%%%%%%%%%%
%Az=0;
%El=0;
%Nt=0;
%T=0;
% alpha=cosd(El)*sind(Az)*cosd(Nt)-cosd(El)*cosd(Az)*sind(Nt);
% beta=-cosd(El)*sind(Az)*sind(Nt)*sind(T)+sind(El)*cosd(T)-cosd(El)*cosd(Az)*cosd(Nt)*sind(T);
%if (Az==0&&El==0)
 %   phi=0;
  %  theta=0;
%else
%theta=acosd(cosd(El)*sind(Az)*sind(Nt)*cosd(T)+sind(El)*sind(T)+cosd(El)*cosd(Az)*cos(Nt)*cosd(T));
%phi=atand((-cosd(El)*sind(Az)*sind(Nt)*sind(T)+sind(El)*cosd(T)-cosd(El)*cosd(Az)*cosd(Nt)*sind(T))/(cosd(El)*sind(Az)*cosd(Nt)-cosd(El)*cosd(Az)*sind(Nt)));
%end

%x0=(2*pi*delta_x*sind(theta)*cosd(phi))/lembda;
%y0=(2*pi*delta_y*sind(theta)*sind(phi))/lembda;

%%%%%%%%%%%-------------Power to each element is 100 Watt-------%%%%%%%%%%
%for y=1:1:m
  %  for x=1:1:n
   %    if (mod((y+x),2)==0)
%             element_power(y,x)=(excitation1)^2;             
%        end
%     end
% end

%%%%%%%%%%%------Uniform Excitation to each element------%%%%%%%%%%%%%%%%%
%for y=1:1:m
    %for x=1:1:n
     %  if (mod((y+x),2)==0)
%             element_excitation(y,x)=10;             
%        end
%     end
% end

%pcolor( element_excitation)
%colorbar
%%%%%%%%%%%----------Taylor excitation----------%%%%%%%%%%%%%%%%%%%%%
%if (excitation==1)
   
    %row_taylor=taylorwin(m,6,-40);
    %col_taylor=(taylorwin(n,6,-30))';
    %taylor_matrix=row_taylor*col_taylor;
    %MAX_taylor=max(max(taylor_matrix));
    %taylor_matrix=(taylor_matrix/MAX_taylor);
    %element_excitation=((taylor_matrix).*(element_excitation)); %excitation of every element
    %element_power=(taylor_matrix).*(element_power);             %power of every element 
%end
%distance= zeros(length_x,length_y,m,n);
%probe_phase = distance;
%probe_power=distance;
%probe_final_vectors=distance;
%probe_amplitude=distance;
%distance_sqr=distance;

%y_offset=(trans_start_y)-(scan_start_y);
%x_offset=(trans_start_x)-(scan_start_x);

%%%%%%%%%%%%%%%%------------Near Field Data calculation-----%%%%%%%%%%%

% %for x=0:1:length_x-1
%  %   for y=0:1:length_y-1
%        for p=0:1:m-1
%             for q=0:1:n-1
%              distance(y+1,x+1,p+1,q+1)= sqrt((d*d)+(y_offset+p*delta_y-y*ystep)*(y_offset+p*delta_y-y*ystep)+(x_offset+q*delta_x-x*xstep)*(x_offset+q*delta_x-x*xstep));
%              probe_phase(y+1,x+1,p+1,q+1)=exp(1j*((x0*q)+ (y0*p)+((2*pi*distance(y+1,x+1,p+1,q+1))/lembda)));
%              distance_sqr(y+1,x+1,p+1,q+1)=((4*pi)^2)*((distance(y+1,x+1,p+1,q+1))^2);
%              probe_power(y+1,x+1,p+1,q+1)=(element_power(p+1,q+1)*lembda*lembda)/(distance_sqr(y+1,x+1,p+1,q+1));    
%             % probe_amplitude(y+1,x+1,p+1,q+1)=sqrt(probe_power(y+1,x+1,p+1,q+1)/2);
%              probe_final_vectors(y+1,x+1,p+1,q+1)=element_excitation(p+1,q+1)*(probe_phase(y+1,x+1,p+1,q+1)/(distance(y+1,x+1,p+1,q+1)));    
%             end
%         end
%     end
% end
   
%%%%%%%---------- power level at different probe positions----%%%%%%%%
% power_level=zeros(length_y,length_x);   

%%%%%%%--------- final near field data at probe positions----%%%%%%
finalNfData=zeros(length_y,length_x);  
fid = fopen('amplitude','r')

while ~feof(fid)
amplitudedata=fscanf(fid,'%d');
end
cnt=1;
x=amplitudedata
row1=x(1:1:65);
row2=x(66:1:131);
row3=x(131:1:196);
row4=x(196:1:260);
row5=x(261:1:325);
row6=x(326:1:390);
row7=x(391:1:455);
row8=x(456:1:520);
row9=x(521:1:585);
row10=x(586:1:650);
row11=x(651:1:715);
row12=x(716:1:780);
row13=x(781:1:845);
row14=x(846:1:910);
row15=x(911:1:975);
row16=x(976:1:1040);
row17=x(1041:1:1105);
row18=x(1106:1:1170);
row19=x(1171:1:1235);
row20=x(1236:1:1300);
row21=x(1301:1:1365);
row22=x(1366:1:1430);
row23=x(1431:1:1495);
row24=x(1496:1:1560);
row25=x(1561:1:1625);
row26=x(1626:1:1690);
row27=x(1691:1:1755);
row28=x(1756:1:1820);
row29=x(1821:1:1885);
row30=x(1886:1:1950);
row31=x(1951:1:2015);
row32=x(2016:1:2080);
row33=x(2081:1:2145);
row34=x(2146:1:2210);
row35=x(2211:1:2275);
row36=x(2276:1:2340);
row37=x(2341:1:2405);
row38=x(2406:1:2470);
row39=x(2471:1:2535);
row40=x(2536:1:2600);
row41=x(2601:1:2665);
row42=x(2666:1:2730);
row43=x(2731:1:2795);
row44=x(2796:1:2860);
row45=x(2861:1:2925);
row46=x(2926:1:2990);
row47=x(2991:1:3055);
row48=x(3056:1:3120);
row49=x(3121:1:3185);
row50=x(3186:1:3250);
row51=x(3251:1:3315);
row52=x(3316:1:3380);
row53=x(3381:1:3445);
row54=x(3446:1:3510);
row55=x(3511:1:3575);
row56=x(3576:1:3640);
row57=x(3641:1:3705);
row58=x(3706:1:3770);
row59=x(3771:1:3835);
row60=x(3836:1:3900);
row61=x(3901:1:3965);
row62=x(3966:1:4030);
row63=x(4031:1:4095);
row64=x(4096:1:4160);
row65=x(4161:1:4225);



% for x=0:1:length_x-1
%     for y=0:1:length_y-1
%         finalNfData(y+1,x+1)=sum(sum(probe_final_vectors(y+1,x+1,:,:)));
%         power_level(y+1,x+1)=sum(sum(probe_power(y+1,x+1,:,:)));
%        %probe_phase(y+1,x+1)=1j*((x0*q)+ (y0*p)+((2*pi*distance(y+1,x+1,p+1,q+1))/lembda))
%     end
% end
%  
x_axis_variation=-255:1:256;
y_axis_variation=-255:1:256;
figure
pcolor(x_span,y_span,10*log10(power_level)+30)
title('Power level (in dBm)')
shading flat
colorbar
 
figure
pcolor((scan_start_x:xstep:scan_stop_x),(scan_start_y:ystep:scan_stop_y),20*log10(abs(finalNfData)))
title('Near Field Pattern')
shading flat
colorbar

figure
pcolor((scan_start_x:xstep:scan_stop_x),(scan_start_y:ystep:scan_stop_y),(angle(finalNfData)*(180/pi)))
title('Near Field Pattern')
shading flat
colorbar


%%%%%%%%%--converting near field to far field pattern-------%%%%%%%%
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

%%%%%%%%-------Azimuth Cut----------%%%%%%%%%%%%%%%%
f5 = figure('NumberTitle','off','Name','Azimuth Probe Compensation Comparision');
plot(Az_Points,20*log10(abs(AF_on_Atlas(el,:))/MAX),'b');
hold on
grid on
title('Azimuth Cut','FontSize',14)
xlim([-90 90]) ;
ylim([-50 0]) ;

% %%%%%%------ElevationProbe Compensation Comparision-------%%%%%%%%%
f6 = figure('NumberTitle','off','Name','ElevationProbe Compensation Comparision');
plot(El_Points,20*log10(abs(AF_on_Atlas(:,az))/MAX),'r');
hold on
grid on
title('Elevation Cut ','FontSize',14)
xlim([-90 90]) ;
ylim([-50 0]) ;
% steps to get elevation cut and azimuth cut
% t=(-255/1.42:1/1.42:256/1.42);
% figure
% subplot(1,2,1)
% plot(t,20*log10(divide(el,:)))
% title('Azimuth cut')
% xlabel('Angles')
% ylabel('Amplitude (in dB)')
% grid on
% 
% subplot(1,2,2)
% plot(t,20*log10(divide(:,az)))
% title('Elevation cut')
% xlabel('Angles')
% ylabel('Amplitude (in dB)')
% grid on
