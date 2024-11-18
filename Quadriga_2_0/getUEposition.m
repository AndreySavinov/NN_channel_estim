function [ XY] = getUEposition( RmaxC,RminC,PhiMaxC,PhiB,PhiShift,Nc,N,Ri )

XY=[];
%generation points in each cluster
for i=1:Nc
  %generation centroid points  
  Phi=rand*(PhiMaxC-2*PhiB)-(PhiMaxC-2*PhiB)/2+PhiShift;
  R=rand*(RmaxC-RminC)+RminC;
  
  Xc=R*cos(Phi);
  Yc=R*sin(Phi);  
  
  if N==0    
    XY=[XY  [Xc;Yc]];
  else
    %generation points in centroid region
    if N(i)==0
      continue
    end
    Phip=rand(1,N(i))*2*pi;
    Rp=rand(1,N(i))*Ri;
    
    X=Rp.*cos(Phip);
    Y=Rp.*sin(Phip);
      
    XY=[XY  [Xc+X;Yc+Y]];
  end
end


end

