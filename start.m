%Rutvij Dhotey
%pattern recognition, CSE583/EE552
%Project 2 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load the WINE data
close all;
clear all;
wine = importdata('./data/wine.data');
%get the class label for each sample
winelabel = wine(:,1);
%each row of winefeature matrix contains the 13 features of each sample
winefeature = wine(:,2:end);

%generate training and test data
train = []; %stores the training samples
test = [];  %stores the test samples
for i=1:3
    ind{i} = find(winelabel==i);
    len = length(ind{i});
    t = randperm(len);
    half = round(len/2);
    train = [train; wine(ind{i}(t(1:half)), :)];
    test = [test; wine(ind{i}(t(half+1:end)), :)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load the FACE data
TrainMat=dlmread('./data/TrainMat.txt');
TestMat=dlmread('./data/TestMat.txt');
LabelTrain=dlmread('./data/LabelTrain.txt');
LabelTest=dlmread('./data/LabelTest.txt');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Part 3: Code for Kmeans Classification Method.
% Assumptions
% In our case we have 3 Clusters. so let numclusters= 3.
% we are calculating for all the dimensions (all 13 Features)

training_dim= length(train (1,:))-1;        %-1 because of the first column.

num_of_data= length(train (:,1));

num_clusters= 3;
%Initialise the Centroids using any one of the three Class rows.
    
   centroid(1,:) = train(6,2:end) ;
   centroid(2,:) = train(31,2:end);
   centroid(3,:) = train(82,2:end);
    
    
% end init centroids
%Calculations and Assignment of data points to each cluster with the
%initialised centroid.

%condition to stop
%while the difference is positive so lets assign the difference to some
%positive value

check_diff= 2. ;
confusionmatrix =[];
iteration=0;

while check_diff>0 

    number=1;
    Assign_Centroid = [];    
    %Calculate the closest centroid to each data point
    for q= 1: length (train(:,1));
         
        %for cluster one
        
        diff_1= (train(q, 2:end) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (train(q,2:end) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
        
    end
    %Confusion Matrix after Every 
    confusionmatrix = [confusionmatrix Assign_Centroid];
       
    %for checking the stopping condition
    old_position_centroid = centroid ;
    
    %calculate new Centroid Position
    centroid = zeros (num_clusters, training_dim);
     pointsInCluster = zeros(num_clusters, 1);

  for d = 1: length(Assign_Centroid);
    centroid(Assign_Centroid(d),:) =centroid(Assign_Centroid(d),:) + train(d,2:end);
    pointsInCluster(Assign_Centroid(d), 1 ) =pointsInCluster(Assign_Centroid(d), 1 ) +1;
    
  end
  
  for c= 1: num_clusters
     if (pointsInCluster(c,1) ~= 0);
        centroid( c , : ) = centroid( c, : ) / pointsInCluster(c, 1);
     else
        centroid(1,:) = train(6,2:end) ;
        centroid(2,:) = train(35,2:end);
        centroid(3,:) = train(83,2:end);
    
     end
      
  end
    
  %stopping
  
  check_diff = sum (sum( (centroid - old_position_centroid).^2 ) );
  
  iteration= iteration+1;

end

  %Data plotting and Confusion Matrix

    final_CM=confusionmatrix(:,iteration);
    final_CM=[final_CM train(:,1)];
    
     final_CM_with4n5dim= [final_CM train(:,5) train(:,11)];
     final_CM_with4n5dim=sortrows(final_CM_with4n5dim,1);

    class14= final_CM_with4n5dim(1:30,3);
    class15= final_CM_with4n5dim(1:30,4);

    class24= final_CM_with4n5dim(31:66,3);
    class25= final_CM_with4n5dim(31:66,4);

    class34= final_CM_with4n5dim(67:90,3);
    class35= final_CM_with4n5dim(67:90,4);

figure(1)

s=scatter (class14,class15);
s.MarkerEdgeColor= 'r';

hold on
s1= scatter (class24,class25);
s1.MarkerEdgeColor= 'b';

hold on
s2=scatter (class34,class35);
s2.MarkerEdgeColor= 'g';

c51x5 = centroid (1,4);
c61x6 = centroid (1,10);
        

c52x5 = centroid (2,4);
c62x6 = centroid (2,10);
       

c53x5 = centroid (3,4);
c63x6 = centroid (3,10);
hold on
plot (c51x5,c61x6,'r*');
hold on       
plot (c52x5,c62x6,'b*');
hold on
plot (c53x5,c63x6,'g*');

%Training data classification


Assign_Centroid = [];    
    

for q= 1: length (train(:,1));
         
        %for cluster one
        
        diff_1= (train(q, 2:end) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (train(q,2:end) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
end
    %Confusion Matrix after Every 
    confusionmatrixtest = Assign_Centroid;
    
    final_CMtest =confusionmatrixtest(:,1);
    final_CMtest =[final_CMtest train(:,1)];
    
    A= zeros (length(train(:,1)),3);
    B= zeros (length(train(:,1)),3);
    
    for i= 1:length(train(:,1))
        if ((final_CMtest(i,1))==1 )
            A(i,1)= 1;
            A(i,2)=0;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==2)
            A(i,1)= 0;
            A(i,2)=1;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==3)
             A(i,1)= 0;
            A(i,2)=0;
            A(i,3)=1;
            end     
        end
        end
    end
    
    for j=1:length(train(:,1))
        if ((final_CMtest(j,2))==1 )
            B(j,1)= 1;
            B(j,2)= 0;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==2)
           B(j,1)= 0;
            B(j,2)= 1;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==3)
            B(j,1)= 0;
            B(j,2)= 0;
            B(j,3)= 1;
            end     
        end
        end
        
    end
        
 figure(2)
 plotconfusion(B',A');
 title('K-Means CONFUSION MATRIX Train');
 
% Testing with Test values

Assign_Centroid = [];    
    

for q= 1: length (test(:,1));
         
        %for cluster one
        
        diff_1= (test(q, 2:end) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (test(q,2:end) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
end
    %Confusion Matrix after Every 
    confusionmatrixtest = Assign_Centroid;
    
    final_CMtest =confusionmatrixtest(:,1);
    final_CMtest =[final_CMtest test(:,1)];
    
    A= zeros (length(test(:,1)),3);
    B= zeros (length(test(:,1)),3);
    
    for i= 1:length(test(:,1))
        if ((final_CMtest(i,1))==1 )
            A(i,1)= 1;
            A(i,2)=0;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==2)
            A(i,1)= 0;
            A(i,2)=1;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==3)
             A(i,1)= 0;
            A(i,2)=0;
            A(i,3)=1;
            end     
        end
        end
    end
    
    for j=1:length(test(:,1))
        if ((final_CMtest(j,2))==1 )
            B(j,1)= 1;
            B(j,2)= 0;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==2)
           B(j,1)= 0;
            B(j,2)= 1;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==3)
            B(j,1)= 0;
            B(j,2)= 0;
            B(j,3)= 1;
            end     
        end
        end
        
    end
        
 figure(3)
 plotconfusion(B',A');
 title('K-Means CONFUSION MATRIX TEST');
 
 %% Part 1 : Least Square Sum Classification 
 
 % 1XK encoding of the target vector. Hence our target vector is now 90x13
 % Dim vector.
 
 target= train (:,1);
 for j=1:length(train(:,1))
        if ((train(j,1))==1 )
            T(j,1)= 1;
            T(j,2)= 0;
            T(j,3)= 0;
        else if ((train(j,1))==2)
           T(j,1)= 0;
            T(j,2)= 1;
            T(j,3)= 0;
        else if ((train(j,1))==3)
            T(j,1)= 0;
            T(j,2)= 0;
            T(j,3)= 1;
            end     
        end
        end
        
 end
 target= T;
 
 %preparing the training matrix with the initial column of 0th order 
 
 Z= ones(num_of_data,1);
 X = [Z train(:,2:end)];
 
 %finding the coefficient vector with 3 class values
 
 Q= X' * X;
 
 Wstar= Q \ X'* target;
 
 %using it on inputs
 %preparing test data
 
  testinput1= [ones(length(train(:,1)),1) train(:,2:end)];
 
 Y = testinput1 * Wstar;
 
 for j=1:length(Y(:,1))
        if (Y(j,1)>Y(j,2) )
                 if (Y(j,1)>Y(j,3))
                    O(j,1)=1;
                    O(j,2)=0;
                    O(j,3)=0;
                else
                    O(j,1)=0;
                    O(j,2)=0;
                    O(j,3)=1;
                end
        else
                if (Y(j,2)>Y(j,3))
                    O(j,1)=0;
                    O(j,2)=1;
                    O(j,3)=0;
                
                else
                    O(j,1)=0;
                    O(j,2)=0;
                    O(j,3)=1;
                end 
        end
 end
 
 output = O;
 
 targettest= train(:,1);
 for j=1:length(train(:,1))
        if ((train(j,1))==1 )
            TT(j,1)= 1;
            TT(j,2)= 0;
            TT(j,3)= 0;
        else if ((train(j,1))==2)
           TT(j,1)= 0;
            TT(j,2)= 1;
            TT(j,3)= 0;
        else if ((train(j,1))==3)
            TT(j,1)= 0;
            TT(j,2)= 0;
            TT(j,3)= 1;
            end     
        end
        end
        
 end
 targettest= TT;
                    
 figure(4)
 plotconfusion(targettest',output');
 title('Least Square CONFUSION MATRIX Train');

 
%Testing
 
 testinput= [ones(length(test(:,1)),1) test(:,2:end)];
 
 Y = testinput * Wstar;
 
 for j=1:length(Y(:,1))
        if (Y(j,1)>Y(j,2) )
                 if (Y(j,1)>Y(j,3))
                    O(j,1)=1;
                    O(j,2)=0;
                    O(j,3)=0;
                else
                    O(j,1)=0;
                    O(j,2)=0;
                    O(j,3)=1;
                end
        else
                if (Y(j,2)>Y(j,3))
                    O(j,1)=0;
                    O(j,2)=1;
                    O(j,3)=0;
                
                else
                    O(j,1)=0;
                    O(j,2)=0;
                    O(j,3)=1;
                end 
        end
 end
 
 output = O;
 
 targettest= test (:,1);
 for j=1:length(test(:,1))
        if ((test(j,1))==1 )
            TT(j,1)= 1;
            TT(j,2)= 0;
            TT(j,3)= 0;
        else if ((test(j,1))==2)
           TT(j,1)= 0;
            TT(j,2)= 1;
            TT(j,3)= 0;
        else if ((test(j,1))==3)
            TT(j,1)= 0;
            TT(j,2)= 0;
            TT(j,3)= 1;
            end     
        end
        end
        
 end
 targettest= TT;
                    
 figure(5)
 plotconfusion(targettest',output');
 title('Least Square CONFUSION MATRIX Test');
 
 %% Part 2 : Fisher Classification 
 
 %separating the classes
 class1 = train (1:30,2:end);
 class2 = train(31:66,2:end);
 class3 = train (67:end,2:end);
 
 %calculating uk for each dimension
%  for i=1:length(class1(1,:))
%     for j=1:length(class1(:,1)) 
%         u1(1,i) = sum(class1(:,i))/length(class1(:,1));
%     end
%  end
%  
%  for i=1:length(class2(1,:))
%     for j=1:length(class2(:,1)) 
%         u2(1,i) = sum(class2(:,i))/length(class2(:,1));
%     end
%  end
%  
%  for i=1:length(class3(1,:))
%     for j=1:length(class3(:,1)) 
%         u3(1,i) = sum(class3(:,i))/length(class3(:,1));
%     end
%  end
%  
u1= mean(class1);
u2= mean(class2);
u3= mean(class3);


s1k=  cov(class1);          
s2k=  cov(class2);
s3k=  cov(class3); 


% %calcualting sk
% 
% for i=1: length(class1(:,1)) 
%     s1k(i,:)=  cov(class1(i,:),u1); 
%     
% end
% 
% %SK1= sum(s1k);
% 
% for i=1: length(class2(:,1)) 
%     s2k(i,:)=  cov(class1(2,:),u2); 
%     
% end
% 
% %SK2= sum(s2k);
% 
% for i=1: length(class3(:,1)) 
%     s3k(i,:)=  cov(class3(i,:),u3); 
%     
% end
% 
% %SK3= sum(s13);


 
%calculating Sw

SW= s1k + s2k + s3k;

%calculating cumulative mean

%u= ((length(class1(:,1))* u1) + (length(class2(:,1))* u2) + (length(class3(:,1))* u3)) / length(train(:,1));
u= mean (train(:,2:end));

%Calculating SB
SB= length(class1(:,1))* ((u1 - u)'* (u1- u )) + length(class2(:,1))* ((u2 - u)'*(u2-u)) + length(class3(:,1))* ((u3 - u)'*(u3-u));

P= inv(SW)* SB;

[X1 X2]= eig(SB,SW,'chol');

%maximum Eigen Values are included in 12 and 13 column. Hence we chose
%those.

Wstarfisher= [X1(:,12) X1(:,13)];

Yfisher1= train(:,2:end) * Wstarfisher;
Ytestfisher = test(:,2:end)* Wstarfisher;

%The new Data set is now of 2 dimensions 
%Classification of this data : 

JJT = knnclassify(Yfisher1,Yfisher1,train(:,1),3);

JJ = knnclassify(Ytestfisher,Yfisher1,train(:,1),3);



%final Classification

 outputfisher= [];
 
 for j=1:length(JJ(:,1))
        if ((JJ(j,1))==1 )
            TT(j,1)= 1;
            TT(j,2)= 0;
            TT(j,3)= 0;
        else if ((JJ(j,1))==2)
           TT(j,1)= 0;
            TT(j,2)= 1;
            TT(j,3)= 0;
        else if ((JJ(j,1))==3)
            TT(j,1)= 0;
            TT(j,2)= 0;
            TT(j,3)= 1;
            end     
        end
        end
        
 end
 outputfisher = TT;
 
 figure(8)
 plotconfusion(targettest',outputfisher');
 title('Fisher CONFUSION MATRIX Test');
 
 outputfisher= [];
 
 for j=1:length(JJT(:,1))
        if ((JJT(j,1))==1 )
            TT(j,1)= 1;
            TT(j,2)= 0;
            TT(j,3)= 0;
        else if ((JJT(j,1))==2)
           TT(j,1)= 0;
            TT(j,2)= 1;
            TT(j,3)= 0;
        else if ((JJT(j,1))==3)
            TT(j,1)= 0;
            TT(j,2)= 0;
            TT(j,3)= 1;
            end     
        end
        end
        
 end
 outputfisher = TT;
 
 targettrain= train (:,1);
 for j=1:length(train(:,1))
        if ((train(j,1))==1 )
            TT(j,1)= 1;
            TT(j,2)= 0;
            TT(j,3)= 0;
        else if ((train(j,1))==2)
           TT(j,1)= 0;
            TT(j,2)= 1;
            TT(j,3)= 0;
        else if ((train(j,1))==3)
            TT(j,1)= 0;
            TT(j,2)= 0;
            TT(j,3)= 1;
            end     
        end
        end
        
 end
 targettrain= TT;
 
 figure(9)
 plotconfusion(targettrain',outputfisher');
 title('Fisher CONFUSION MATRIX Train');
 
 
 
 
 
%% Extra Credit

%% Part 4: Kmeans on Yfisher1

% Assumptions
% In our case we have 3 Clusters. so let numclusters= 3.
% we are calculating for all the dimensions (all 13 Features)

training_dim= length(Yfisher1 (1,:));        %-1 because of the first column.

num_of_data= length(Yfisher1 (:,1));

num_clusters= 3;
%Initialise the Centroids using any one of the three Class rows.
   
    centroid=[];
   centroid(1,:) = Yfisher1(6,:) ;
   centroid(2,:) = Yfisher1(31,:);
   centroid(3,:) = Yfisher1(82,:);
    
    
% end init centroids
%Calculations and Assignment of data points to each cluster with the
%initialised centroid.

%condition to stop
%while the difference is positive so lets assign the difference to some
%positive value

check_diff= 2. ;
confusionmatrix =[];
iteration=0;

while check_diff>0 

    number=1;
    Assign_Centroid = [];    
    %Calculate the closest centroid to each data point
    for q= 1: length (Yfisher1(:,1));
         
        %for cluster one
        
        diff_1= (Yfisher1(q,:) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (Yfisher1(q,:) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
        
    end
    %Confusion Matrix after Every 
    confusionmatrix = [confusionmatrix Assign_Centroid];
       
    %for checking the stopping condition
    old_position_centroid = centroid ;
    
    %calculate new Centroid Position
    centroid = zeros (num_clusters, training_dim);
     pointsInCluster = zeros(num_clusters, 1);

  for d = 1: length(Assign_Centroid);
    centroid(Assign_Centroid(d),:) =centroid(Assign_Centroid(d),:) + Yfisher1(d,:);
    pointsInCluster(Assign_Centroid(d), 1 ) =pointsInCluster(Assign_Centroid(d), 1 ) +1;
    
  end
  
  for c= 1: num_clusters
     if (pointsInCluster(c,1) ~= 0);
        centroid( c , : ) = centroid( c, : ) / pointsInCluster(c, 1);
     else
        centroid(1,:) = Yfisher1(6,:) ;
        centroid(2,:) = Yfisher1(35,:);
        centroid(3,:) = Yfisher1(83,:);
    
     end
      
  end
    
  %stopping
  
  check_diff = sum (sum( (centroid - old_position_centroid).^2 ) );
  
  iteration= iteration+1;

end

%Training data classification


Assign_Centroid = [];    
    

for q= 1: length (Yfisher1(:,1));
         
        %for cluster one
        
        diff_1= (Yfisher1(q, :) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (Yfisher1(q,:) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
end
    %Confusion Matrix after Every 
    confusionmatrixtest = Assign_Centroid;
    
    final_CMtest =confusionmatrixtest(:,1);
    final_CMtest =[final_CMtest Yfisher1(:,1)];
    
    A= zeros (length(Yfisher1(:,1)),3);
    B= zeros (length(Yfisher1(:,1)),3);
    
    for i= 1:length(Yfisher1(:,1))
        if ((final_CMtest(i,1))==1 )
            A(i,1)= 1;
            A(i,2)=0;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==2)
            A(i,1)= 0;
            A(i,2)=1;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==3)
             A(i,1)= 0;
            A(i,2)=0;
            A(i,3)=1;
            end     
        end
        end
    end
    
    for j=1:length(Yfisher1(:,1))
        if ((final_CMtest(j,2))==1 )
            B(j,1)= 1;
            B(j,2)= 0;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==2)
           B(j,1)= 0;
            B(j,2)= 1;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==3)
            B(j,1)= 0;
            B(j,2)= 0;
            B(j,3)= 1;
            end     
        end
        end
        
    end
        
 %figure(10)
 %plotconfusion(B',A');
 %title('K-Means CONFUSION MATRIX Train FISHER');
 
% Testing with Test values

Assign_Centroid = [];    
    

for q= 1: length (Ytestfisher(:,1));
         
        %for cluster one
        
        diff_1= (Ytestfisher(q, :) - centroid(1,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 1;
        
        %for other clusters
        for c= 2: num_clusters;
            
            
            diff_to_centroid= (Ytestfisher(q,:) - centroid (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        end
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
end
    %Confusion Matrix after Every 
    confusionmatrixtest = Assign_Centroid;
    
    final_CMtest =confusionmatrixtest(:,1);
    final_CMtest =[final_CMtest Ytestfisher(:,1)];
    
    A= zeros (length(Ytestfisher(:,1)),3);
    B= zeros (length(Ytestfisher(:,1)),3);
    
    for i= 1:length(Ytestfisher(:,1))
        if ((final_CMtest(i,1))==1 )
            A(i,1)= 1;
            A(i,2)=0;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==2)
            A(i,1)= 0;
            A(i,2)=1;
            A(i,3)=0;
        else if ((final_CMtest(i,1))==3)
             A(i,1)= 0;
            A(i,2)=0;
            A(i,3)=1;
            end     
        end
        end
    end
    
    for j=1:length(Ytestfisher(:,1))
        if ((final_CMtest(j,2))==1 )
            B(j,1)= 1;
            B(j,2)= 0;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==2)
           B(j,1)= 0;
            B(j,2)= 1;
            B(j,3)= 0;
        else if ((final_CMtest(j,2))==3)
            B(j,1)= 0;
            B(j,2)= 0;
            B(j,3)= 1;
            end     
        end
        end
        
    end
        
%figure(11)
%plotconfusion(B',A');
%title('K-Means CONFUSION MATRIX TEST FISHER');
 






 %% Part 3a : Least Square Sum Classification for FACE DETECTION
 
 % 1XK encoding of the target vector. Hence our target vector is now 90x13
 % Dim vector.
 
 targetFACE= LabelTrain (:,1);
 for j=1:length(LabelTrain(:,1))
        if ((LabelTrain(j,1))==1 )
            Tf(j,1)= 1;
            Tf(j,2)= 0;
            
        else 
            Tf(j,1)= 0;
            Tf(j,2)= 1;
            
        end
        
 end
 targetFACE= Tf;
 
 %preparing the training matrix with the initial column of 0th order 
 
 Zf = ones(length(TrainMat(:,1)),1);
 Xf = [Zf TrainMat];
 
 %finding the coefficient vector with 3 class values
 
 Qf = Xf' * Xf;
 
 Wstarface= Qf \ Xf' * targetFACE;
 
 %using it on inputs
 %preparing test data
 
 testinputface= [ones(length(TestMat(:,1)),1) TestMat];
 
 Yface = testinputface * Wstarface;
 
 for j=1:length(Yface(:,1))
        if (Yface(j,1)>Yface(j,2) )
                    Oface(j,1)=1;
                    Oface(j,2)=0;
                   
                else
                    Oface(j,1)=0;
                    Oface(j,2)=1;
               
               
        end
 end
 
 
 
 targettestface= LabelTest (:,1);
 for j=1:length(LabelTest(:,1))
        if ((LabelTest(j,1))==1 )
            TTface(j,1)= 1;
            TTface(j,2)= 0;
            
        else
            TTface(j,1)= 0;
            TTface(j,2)= 1;
            
        end
        
 end
 targettestface= TTface;
                    
 figure(6)
 plotconfusion(targettestface',Oface');
 title('Least Square FACE : CONFUSION MATRIX');
 
 %% Part 3b : Kmeans Classification for FACE DETECTION 
% Assumptions
% In our case we have 3 Clusters. so let numclusters= 3.
% we are calculating for all the dimensions (all 13 Features)

training_dim= length(TrainMat (1,:));        %-1 because of the first column.

num_of_data= length(TrainMat (:,1));

num_clusters= 2;
%Initialise the Centroids using any one of the three Class rows.
    
   centroidf(2,:) = TrainMat(6,:) ;
   centroidf(1,:) = TrainMat(45,:);
    
    
% end init centroids
%Calculations and Assignment of data points to each cluster with the
%initialised centroid.

%condition to stop
%while the difference is positive so lets assign the difference to some
%positive value

check_diff= 2. ;
confusionmatrix =[];
iteration=0;

while check_diff>0 

    number=1;
    Assign_Centroid = [];    
    %Calculate the closest centroid to each data point
    for q= 1: length (TrainMat(:,1));
         
        %for cluster one
        
        diff_1= (TrainMat(q, :) - centroidf(2,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 2;
        
        %for other clusters
         c= 1;
            
            
            diff_to_centroid= (TrainMat(q,:) - centroidf (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
        
        
    end
    %Confusion Matrix after Every 
    confusionmatrix = [confusionmatrix Assign_Centroid];
       
    %for checking the stopping condition
    old_position_centroid = centroidf ;
    
    %calculate new Centroid Position
    centroidf = zeros (num_clusters, training_dim);
     pointsInCluster = zeros(num_clusters, 1);

  for d = 1: length(Assign_Centroid);
    centroidf(Assign_Centroid(d),:) =centroidf(Assign_Centroid(d),:) + TrainMat(d,:);
    pointsInCluster(Assign_Centroid(d), 1 ) =pointsInCluster(Assign_Centroid(d), 1 ) +1;
    
  end
  
  for c= 1: num_clusters
     if (pointsInCluster(c,1) ~= 0);
        centroidf( c , : ) = centroidf( c, : ) / pointsInCluster(c, 1);
     else
        centroidf(1,:) = TrainMat(1,:) ;
        centroidf(2,:) = TrainMat(50,:);
        
     end
      
  end
    
  %stopping
  
  check_diff = sum (sum( (centroidf - old_position_centroid).^2 ) );
  
  iteration= iteration+1;

end

  %Data plotting and Confusion Matrix

    final_CM=confusionmatrix(:,iteration);
    final_CM=[final_CM TrainMat(:,1)];
    
   
%% Testing with Test values

Assign_Centroid = [];    
    

for q= 1: length (TestMat(:,1));
         
        %for cluster one
        
        diff_1= (TestMat(q, :) - centroidf(2,:));
        diff_1= diff_1 * (diff_1)';
        current_assignment_cluster= 2;
        
        %for other clusters
         c= 1;
            
            
            diff_to_centroid= (TestMat(q,:) - centroidf (c, :));
            diff_to_centroid= diff_to_centroid * diff_to_centroid';
           
            % assignment 
            if (diff_1 > diff_to_centroid )
                current_assignment_cluster= c;
                diff_1= diff_to_centroid;     
            end
        
        
        Assign_Centroid= [Assign_Centroid ; current_assignment_cluster];
        
end    
        

    %Confusion Matrix after Every 
    confusionmatrixtest = Assign_Centroid;
    
    final_CMtest =confusionmatrixtest(:,1);
    final_CMtest =[final_CMtest LabelTest];
    
    A= zeros (length(TestMat(:,1)),2);
    B= zeros (length(TestMat(:,1)),2);
    
    for i= 1:length(TestMat(:,1))
        if ((final_CMtest(i,1))==1 )
            A(i,1)= 1;
            A(i,2)=0;
            
        else 
            A(i,1)= 0;
            A(i,2)=1;
            
        end
    end
    
    for j=1:length(LabelTest(:,1))
        if ((final_CMtest(j,2))==1 )
            B(j,1)= 1;
            B(j,2)= 0;
            
        else
           B(j,1)= 0;
            B(j,2)= 1;
            
        end
        
    end
 
 %A= [A(:,2) A(:,1)];
    
 figure(7)
 plotconfusion(B',A');
 title('K-Means FACE: CONFUSION MATRIX'); 
 
%% END





