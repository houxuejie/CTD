function action = func_epsilon_greedy_action(state,q_table,A,destination,epsilon)
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


rand_number=rand;
actions=find(A(state,:)==1);%�ҵ���ǰ״̬����ִ�е�action
num_action=length(actions);
if state==destination
    %% change a little with meaning
    action=destination-1;
else
    if rand_number>=epsilon
        get_action_qvalue=zeros(1,num_action);
        for i=1:num_action
            get_action_qvalue(i)=q_table(state,actions(i));
        end
        [~,index]=min(get_action_qvalue);
        action=actions(index);
    else
        index=unidrnd(num_action,1);
        action=actions(index);
    end
end
end

