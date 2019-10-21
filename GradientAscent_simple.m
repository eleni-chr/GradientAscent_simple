function GradientAscent_simple
%% NOTE: This function was written using MATLAB R2016a

% NOTE: This function uses a SIMPLE landscape. Another version of this
% function, using a more COMPLEX landscape can be found on GitHub, titled
% "GradientAscent_complex".

% This algorithm contains the function SimpleLandscape which takes an (x,y)
% position as input and returns the height vector of a function of x and y.
% It also contains the function SimpleLandscapeGrad which takes the same
% (x,y) position as input and returns the gradient vector of a function of
% x and y. With respect to the three variables in this algorithm, the
% function will initially:
% 1. use a constant learning rate of 0.1;
% 2. stop after a maximum of 50 iterations of the algorithm;
% 3. as the functions being optimised are known, the exact gradient will be used.
% To perform gradient ascent, the function first calculates the gradient at
% a point and then uses this gradient to update the point it is at. The
% function SimpleLandscapeGrad returns the elements of the gradient of the
% corresponding function as a vector.

%% Plot the landscape
ezmesh(@SimpleLandscape,[-2 2],[-2 2])

% Enter maximum number of iterations of the algorithm, learning rate and mutation range
% Changing these parameters might improve/worsen the performance of the
% algorithm. Try it and see.
NumSteps=50;
LRate=0.1;

% Choose a random starting point with x and y in the range (-2, 2) or to start from a grid of points (un/comment relevant line)
StartPt=-2+4.*rand(1,2);
%StartPt=point_grid(-2,2,-2,2,0.5);

% Define how many starting points there are
a=size(StartPt); NumPts=a(1,1); %NumPts is equal to the number of starting points

% Find maximum
GradAscent(StartPt,NumSteps,LRate,NumPts);

end

% Function implementing gradient ascent
function GradAscent(StartPt,NumSteps,LRate,NumPts)
    steps=zeros(1,NumPts); %A zero matrix in which the steps taken by each StartPt will be stored in for later use
    test=false; %test is used later to store the step number in the matrix 'steps'
    breakloop=false; %breakloop is used later to break out of the outermost for-loop
	PauseFlag=1; %PauseFlag is used later to pause to view the output
    heights = zeros(NumSteps,NumPts); %A vector of zeros in which the height of each StartPt will be stored in for later use
	hold on;
	for i = 1:NumSteps
        for j = 1:NumPts
	    % Calculate the 'height' at StartPt
        h = SimpleLandscape(StartPt(j,1),StartPt(j,2));
        
        % Store the height of each StartPt in 'heights' at each iteration for later use
        heights(i,j)=h;
    
	    % Plot point(s) on the landscape
        plot3(StartPt(j,1),StartPt(j,2),h,'*','MarkerSize',10)
	
	    % Calculate the gradient at each StartPt
        g = SimpleLandscapeGrad(StartPt(j,1),StartPt(j,2));
        
	    % Calculate the new point and update each StartPt
        StartPt(j,:) = StartPt(j,:) + LRate.*g;
        
        %Check if global optimum is reached and at which step, then stop algorithm if global optimum is reached before the final NumSteps
        if h==4 %hard-coded because we know the max value by looking at the plot and the function
            optimum_reached = true %returns a 1 if global optimum is reached
            steps_taken = i %returns at which step the global optimum was reached
            test=true; %test is updated to true and is used later to store the step numbers in the matrix steps
            %Note: un-comment the following two lines to stop the algorithm as soon as one point finds the global maximum
            %breakloop=true; %breakloop is updated to true and is used later to break out of the outermost for-loop
            %break %breaks out of the innermost for-loop
        end     
        if (test)
            steps(1,j)=i+1; %steps at those NumPts where h==4 is equal to the number of steps taken to find max
        end
         
        % Ensure each StartPt is within the specified bounds
	    StartPt(j,:) = max([StartPt(j,:);-2 -2]);
	    StartPt(j,:) = min([StartPt(j,:);2 2]);
        end
        
        if (breakloop)
            break %breaks out of the outermost for-loop (i.e., the GradAscent algorithm stops)
        end
        
        % Pause to view output
	    if(PauseFlag)
	        x=input('Press return to continue\nor 0 and return to stop pausing\n');
	        if(x==0)
                PauseFlag=0;
            end
        end
    end
    if max(heights)~=4
        optimum_reached = false %returns 0 if the global optimum is not reached after all steps are completed
        steps_taken = NumSteps %returns the number of the last step
    end
	hold off

    %Plot a pseudocolour plot
    A=logical(heights==4); %returns ones at column positions that correspond to starting points which reached the maximum
    B=sum(A); %sums all elements in each column of matrix A
    if NumPts>1
        steps(B==0)=inf;
        Col=steps-B;
        Colours=vec2mat(Col,sqrt(NumPts)); %The values of the elements of Colours specify the color in each cell of figure 2
        figure(2) %a pseudocolour plot with the darkest cells indicating points that reached the maximum with the least number of steps
        pcolor(Colours); axis off; colormap(copper); shading faceted; h=colorbar; set(h,'FontSize',12); title('Gradient Ascent','FontSize',12);
        m=max(heights); %finds the maximum value of h reached by each of the starting points
        Colours2=vec2mat(m,sqrt(NumPts)); %The values of the elements of Colours2 specify the color in each cell of figure 3
        figure(3) %a pseudocolour plot with the darkest cells indicating points that reached a larger height
        pcolor(Colours2); axis off; colormap(flipud(bone)); shading faceted; h=colorbar; set(h,'FontSize',12); title('Gradient Ascent','FontSize',12);
     end

    %Return how many times the global maximum was found
    C=B(B~=0); %B contains some columns of only zeroes that correspond to the starting points that never reached the maximum. This line creates a new matrix C which does not contain those zero columns. The number of columns in C is equal to the number of starting points that reached the maximum
    D=size(C);
    successful_points=D(1,2) %how many starting points reached the maximum
    steps(B==0)=0;
    count=steps-B; %how many steps it took each point to reach the maximum (includes zeroes for those points that never reached the maximum)
    keepers=count(count~=0); %removes the zeroes from count
    %Note: lines 57-58 must be commented out for the following if-statement to work properly
    if (isempty(keepers)) %if this statement is true it means that the max was not found, so mean_steps should be 0
        mean_steps=0
    else
        mean_steps=mean(keepers) %returns the mean number of steps taken by each StartPt to find the maximum
    end
end

% Definition of Simple landscape
function [z] = SimpleLandscape(x,y)
	z=max(1-abs(2*x),0)+x+y;
end

% Definition of gradient of Simple landscape
function [g] = SimpleLandscapeGrad(x,y)
	if(1-abs(2*x) > 0)
	    if(x<0) g(1) = 3;
	    elseif(x==0) g(1)=0;
	    else g(1) = -1;
	    end
	else g(1)=1;
	end
	g(2)=1;
end

%Function creating a grid of points on the landscape
function M = point_grid(minx,maxx,miny,maxy,step)
x=minx;
y=miny;
M=zeros(((maxx-miny)./step).^2,2);
i=1;
while x<=maxx
    y=miny;
    while y<=maxy
        M(i,:)=[x y];
        i=i+1;
        y=y+step;
    end
    x=x+step;
end
end