function [XY] = getUExy( Nue,Nc,NumSec,UeGeneration,BSxy,RmaxC,RminC,PhiB,Ri )
%---------------------------------------------------------------------------------------------------
%NumSec=3;
NumBS=length(BSxy(1,:));
NUE_sec=zeros(1,NumBS*NumSec);
NUE_sec(1:end)=floor(Nue/(NumSec*NumBS)); 
Oth=Nue-NUE_sec(1)*NumBS*NumSec;
NUE_sec(1:Oth)=NUE_sec(1:Oth)+1;
XY=[];

%---------------------------------------------------------------------------------------------------

switch UeGeneration
  case 'By-Site'
        
    NUE_sec=zeros(1,NumBS);
    NUE_sec(1:end)=floor(Nue/NumBS);
    Oth=Nue-NUE_sec(1)*NumBS;
    NUE_sec(1:Oth)=NUE_sec(1:Oth)+1;
    
    N=0;
    PhiMaxC=360/180*pi;
    PhiB=0/180*pi; 
    PhiShift=0/180*pi;
    Ri=0;
    
    for ii=1:NumBS
      if (NUE_sec(ii)==0)
        continue
      end
      XYc = getUEposition(RmaxC,RminC,PhiMaxC,PhiB,PhiShift,NUE_sec(ii),N,Ri)+repmat([BSxy(1,ii);BSxy(2,ii)],1,NUE_sec(ii));
      XY=[XY XYc];      
    end
    
    
  case 'By-Cell'
    
    PhiMaxC=120/180*pi;   
    N=0;
    Ri=0;
    
    for iBS=1:NumBS
      for iSEC=1:NumSec
        if (NUE_sec(NumSec*(iBS-1)+iSEC)==0)
          continue
        end
        PhiShift=((iSEC-1)*120+30)/180*pi;        
        XYc = getUEposition(RmaxC,RminC,PhiMaxC,PhiB,PhiShift,NUE_sec(NumSec*(iBS-1)+iSEC),N,Ri)+repmat([BSxy(1,iBS);BSxy(2,iBS)],1,NUE_sec(NumSec*(iBS-1)+iSEC));
        XY=[XY XYc];      
      end    
    end
    
          
  case 'By-Cell-Centroids'
    
    PhiMaxC=120/180*pi;
    NUE_sec_c=zeros(Nc,NumBS*NumSec);
    NUE_sec_c(:,1:end)=floor(Nue/(NumSec*NumBS*Nc)); 
    Oth=Nue-NUE_sec_c(1)*NumBS*NumSec*Nc;
    NUE_sec_c(1:Oth)=NUE_sec_c(1:Oth)+1; 
    
    for iBS=1:NumBS
      for iSEC=1:NumSec
%         if (sum(NUE_sec_c(:,NumSec*(iBS-1)+iSEC))==0)
%           continue
%         end
        PhiShift=((iSEC-1)*120+30)/180*pi;     
        XYc = getUEposition(RmaxC,RminC,PhiMaxC,PhiB,PhiShift,Nc,NUE_sec_c(:,NumSec*(iBS-1)+iSEC),Ri)+repmat([BSxy(1,iBS);BSxy(2,iBS)],1,sum(NUE_sec_c(:,NumSec*(iBS-1)+iSEC)));
        XY=[XY XYc];      
      end
    end    
        
  otherwise
      error('Not specified method of UE generation! Check settings of simulation.');
end



%--------------------------------------- pllots UE positions ---------------------------------------
if (0)
  scatter(XY(1,:),XY(2,:),'fill');
  grid on  
end






end

