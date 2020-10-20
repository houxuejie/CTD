clear ;
clc;
close all;

%% input map,orgin and destination,all links' mean value and std of travel time
destination = 2;
start_node=ones(1,4);
end_node=start_node+1;
A=zeros(2,4);
for i=1:4
    A(start_node(i),i)=1;
end
for i=1:4
    A(end_node(i),i)=-1;
end
[num_states,num_actions]=size(A);%generate the number of states and actions


mean_value=zeros(num_actions,1);
for i=1:num_actions
    mean_value(i)=1*i;
end

std_value=zeros(num_actions,1);
for i=1:num_actions
    coff1=4/3;
    coff2=2/3;
    std_value(i)=(coff1/i)+coff2;
end
covSigma=zeros(num_actions);

for i=1:num_actions
    covSigma(i,i)=std_value(i)^2;
end

origin = 1;
b = zeros(num_states,1);
b(origin) = 1;
b(destination) = -1;
%% two Q tables are established to store mean and variance value
q_mean_table_initial=zeros(num_states,num_actions);
q_variance_table_initial=zeros(num_states,num_actions);
% the combination of mean value and std value
zeta=1;
q_table=q_mean_table_initial+zeta*sqrt(q_variance_table_initial);


%相关参数
epsilon=0.5;
gamma=1;
episode_max=40000;

number=0;
average_number=100;
interval_num=10;
performance_TD2=zeros(fix(episode_max/interval_num),1);
performance_TD2_total=zeros(fix(episode_max/interval_num),1);
q_mean_table_total=zeros(episode_max,4);
q_mean_table_count=zeros(episode_max,4);
q_mean_table_countlink1=zeros(episode_max,average_number);
q_mean_table_countlink2=zeros(episode_max,average_number);
q_mean_table_countlink3=zeros(episode_max,average_number);
q_mean_table_countlink4=zeros(episode_max,average_number);
q_variance_table_countlink1=zeros(episode_max,average_number);
q_variance_table_countlink2=zeros(episode_max,average_number);
q_variance_table_countlink3=zeros(episode_max,average_number);
q_variance_table_countlink4=zeros(episode_max,average_number);
for i=1:average_number
    number=0;
    q_mean_table=q_mean_table_initial;
    q_variance_table=q_variance_table_initial;
    episode_count=1;
    count_action=zeros(4,1);
    while episode_count<=episode_max
        %% Storing data
        q_mean_table_count(episode_count,:)=q_mean_table(1,:);
        q_mean_table_countlink1(episode_count,i)=q_mean_table(1,1);
        q_mean_table_countlink2(episode_count,i)=q_mean_table(1,2);
        q_mean_table_countlink3(episode_count,i)=q_mean_table(1,3);
        q_mean_table_countlink4(episode_count,i)=q_mean_table(1,4);
        q_variance_table_countlink1(episode_count,i)=q_variance_table(1,1);
        q_variance_table_countlink2(episode_count,i)=q_variance_table(1,2);
        q_variance_table_countlink3(episode_count,i)=q_variance_table(1,3);
        q_variance_table_countlink4(episode_count,i)=q_variance_table(1,4);
        %start node
        state=1;
        %generate an action base on epsilon-greedy
        action = func_epsilon_greedy_action(state,q_table,A,destination,epsilon);
        count_action(action)=count_action(action)+1;
        alpha=0.01;
        alpha_variance=0.001;
        while state~=destination

            %Get next_ state according to state and action
            next_state = func_next_state(state,action,A,destination);
            %generate next_action base on epsilon-greedy
            next_action = func_epsilon_greedy_action(next_state,q_table,A,destination,epsilon);
           %% calculate TD error
            %reward represents the travel time of the path, which is generated randomly according to four different random processes, and must be positive
            reward = func_reward_four_distribution(state,action,mean_value,std_value,destination);
            %reward = func_reward(state,action,mean_value,std_value,destination);
            delta=reward-q_mean_table(state,action);
            %update Q mean value table
            q_mean_table(state,action)=q_mean_table(state,action)+alpha*delta;

            %update Q variance value table

            R_variance=delta^2;
            delta_variance=R_variance-q_variance_table(state,action);
            q_variance_table(state,action)=q_variance_table(state,action)+alpha_variance*delta_variance;
            % update Q table
            q_table = q_mean_table+zeta*sqrt(q_variance_table);
            state = next_state;
            action = next_action;
            if state==destination
                episode_count=episode_count+1;
                if mod(episode_count,interval_num)==0
                    %% find the path
                    number=number+1;
                    [~,path]=min(q_table(1,:));
                    path_name=zeros(4,1);
                    path_name(path)=1;
                    performance_TD2(number)=mean_value'*path_name+zeta*sqrt(path_name'*covSigma*path_name);
                    break
                end
            end
         end
     end
    q_mean_table_total=q_mean_table_total+q_mean_table_count;

end
q_mean_table_average=q_mean_table_total/average_number;

q_mean_table_mean=zeros(episode_max,4);
q_mean_table_std=zeros(episode_max,4);
q_variance_table_mean=zeros(episode_max,4);
q_variance_table_std=zeros(episode_max,4);
for i=1:episode_max
    q_mean_table_mean(i,1)=mean(q_mean_table_countlink1(i,:));
    q_mean_table_mean(i,2)=mean(q_mean_table_countlink2(i,:));
    q_mean_table_mean(i,3)=mean(q_mean_table_countlink3(i,:));
    q_mean_table_mean(i,4)=mean(q_mean_table_countlink4(i,:));
    q_mean_table_std(i,1)=std(q_mean_table_countlink1(i,:));
    q_mean_table_std(i,2)=std(q_mean_table_countlink2(i,:));
    q_mean_table_std(i,3)=std(q_mean_table_countlink3(i,:));
    q_mean_table_std(i,4)=std(q_mean_table_countlink4(i,:));
    q_variance_table_mean(i,1)=mean(q_variance_table_countlink1(i,:));
    q_variance_table_mean(i,2)=mean(q_variance_table_countlink2(i,:));
    q_variance_table_mean(i,3)=mean(q_variance_table_countlink3(i,:));
    q_variance_table_mean(i,4)=mean(q_variance_table_countlink4(i,:));
    q_variance_table_std(i,1)=std(q_variance_table_countlink1(i,:));
    q_variance_table_std(i,2)=std(q_variance_table_countlink2(i,:));
    q_variance_table_std(i,3)=std(q_variance_table_countlink3(i,:));
    q_variance_table_std(i,4)=std(q_variance_table_countlink4(i,:));
end