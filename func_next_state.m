function next_state = func_next_state(state,action,A,destination)
%next_state”Îaction∂‘”¶



if state==destination
    next_state=state;
else
    next_state=find(A(:,action)==-1);
end
end

