% When committed to a choice of subassembly template (indexed (S,I)) for a node N,
% may commit to a possible part or subassembly options; commit to the sole option
% if there's only one
{ commit_choice(n(N,NS),C) } :- commit_template(N,S,I), node_possible_choice(S,I,NS,C).
commit_choice(n(N,NS),C) :- commit_template(N,S,I), node_sole_choice(S,I,NS,C).
node_choice(S,I,NS,C) :- node_possible_choice(S,I,NS,C).
node_choice(S,I,NS,C) :- node_sole_choice(S,I,NS,C).

% Top node (0) represents the build target subassembly
commit_choice(0,S) :- build_target(S).

% When committed to a choice of subassembly S for the node N, may commit to a
% possible template (S,I); commit to the sole option if there's only one
{ commit_template(N,S,I) } :- commit_choice(N,S), sa_possible_template(S,I).
commit_template(N,S,I) :- commit_choice(N,S), sa_sole_template(S,I).
sa_template(S,I) :- sa_possible_template(S,I).
sa_template(S,I) :- sa_sole_template(S,I).

% Recognize new nodes (as determined by committing to a template (S,I)) mapped by
% skolem functions, while annotating node type
atomic_node(n(N,NS)) :- commit_template(N,S,I), node_choice(S,I,NS,C), atomic(C).
sa_node(n(N,NS)) :- commit_template(N,S,I), node_choice(S,I,NS,C), subassembly(C).
node(N) :- atomic_node(N). node(N) :- sa_node(N).

% Each node has to commit to one and only choice
:- node(N), #count { C : commit_choice(N,C) } != 1.
% Each subassembly-type node has to commit to one and only template
:- sa_node(N), commit_choice(N,S), #count { I : commit_template(N,S,I) } != 1.

% An object that may be an instance of atomic part type P can be considered and
% used as one, but an object cannot be used as more than one atomic part type
% at the same time
{ use_as(O,P) } :- is_likely(O,P,_).
:- is_likely(O,_,_), #count { P : use_as(O,P) } > 1.

% Usage should not exceed demand
part_demand(P,DM) :- atomic(P), DM = #count { N : commit_choice(N,P) }.
part_usage(P,PU) :- atomic(P), PU = #count { O : use_as(O,P) }.
:- part_demand(P,DM), part_usage(P, PU), PU > DM.

% Do not allow any idle objects when they can be employed
unmet_demand(P,DM-PU) :- part_demand(P,DM), part_usage(P, PU).
ununsed_supply(P,US) :- atomic(P),
    US = #count { O : is_likely(O,P,_), not use_as(O,_) }.
:- unmet_demand(P,UD), ununsed_supply(P,US), UD > 0, US > 0.

% For tracking which nodes will have to be connected at which contact points
connect(D1,D2,CP1,CP2) :-
    commit_template(N,S,I),
    fits_signature(D1,N,NS1,SG1), fits_signature(D2,N,NS2,SG2),
    connection_signature(S,I,NS1,NS2,SG1,SG2,CP1,CP2).
fits_signature(n(N,NS),N,NS,P) :- commit_choice(n(N,NS),P), atomic(P).
fits_signature(D,N,NS,c((S,I),SG)) :-
    fits_signature(D,n(N,NS),_,SG), commit_template(n(N,NS),S,I).

% Compute final compatibility score by taking average across all atomic nodes
total_score(TS) :- TS = #sum { X,O : is_likely(O,P,X), use_as(O,P) }.
avg_score(TS/NN) :- total_score(TS), NN = #count { N : atomic_node(N) }, NN != 0.

% Maximize final compatibility score
#maximize { AS : avg_score(AS) }.

% % Toy example
% build_target(0).
% sa_sole_template(0,0).
% node_sole_choice(0,0,0,2).
% node_possible_choice(0,0,1,1). node_possible_choice(0,0,1,5).
% node_sole_choice(0,0,2,3).
% sa_possible_template(3,0).
% node_sole_choice(3,0,0,4).
% node_possible_choice(3,0,1,5). node_possible_choice(3,0,1,1).
% sa_possible_template(3,1).
% node_sole_choice(3,1,0,6).
% connection_signature(0,0,0,1,2,1,p21,p12).
% connection_signature(0,0,0,1,2,5,p25,p52).
% connection_signature(0,0,0,2,2,c((3,0),4),p24,p42).
% connection_signature(0,0,0,2,2,c((3,1),6),p26,p62).
% is_likely(a,1,18).
% is_likely(b,2,16).
% is_likely(c,4,10).
% is_likely(d,5,16).
% is_likely(e,6,10).
% is_likely(f,6,20).
% atomic(1). atomic(2). atomic(4). atomic(5). atomic(6).
% subassembly(0). subassembly(3).