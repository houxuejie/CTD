function reward = func_reward_four_distribution(state,action,mean_value,std_value,destination)

%路径的均值和标准差

if state==destination
    reward=0;
else
    if action==1
        reward=gamrnd(0.25,4);
    elseif action==2
        reward=lognrnd(0.51,0.604);
    elseif action==3       
        reward=normrnd(3,std_value(action));
    elseif action==4
        reward=mvnrnd(mean_value(action),std_value(action));
    else
%         reward=RandomVariate[StableDistribution[0,alpha,beta,mean_value(action),std_value(action)],1]
%         makedist('Stable','alpha',0.5,'beta',1,'gam',1,'delta',0)';
        prop=[0.5 0.5];%prop is the proportion of two-component Gaussian distribution
        mu=[mean_value(action)-1,mean_value(action)+1]';
        sigma=std_value(action);%cat(1,[(std_value(action)-1)^2 0],[0 (std_value(action)+1)^2]);
        gm = gmdistribution(mu,sigma,prop);
        reward=random(gm,1);
    end
end

end
