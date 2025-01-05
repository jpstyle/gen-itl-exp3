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

% Do not allow any idle objects when they can be mployed
unmet_demand(P,DM-PU) :- part_demand(P,DM), part_usage(P, PU).
ununsed_supply(P,US) :- atomic(P),
    US = #count { O : is_likely(O,P,_), not use_as(O,_) }.
:- unmet_demand(P,UD), ununsed_supply(P,US), UD > 0, US > 0.

% For tracking which nodes will have to be connected at which contact points
to_connect(D1,D2,CP1,CP2) :-
    commit_template(N,S,I),
    fits_signature(D1,N,NS1,SG1), fits_signature(D2,N,NS2,SG2),
    connection_signature(S,I,NS1,NS2,SG1,SG2,CP1,CP2).
fits_signature(n(N,NS),N,NS,P) :- commit_choice(n(N,NS),P), atomic(P).
fits_signature(D,N,NS,c((S,I),SG)) :-
    fits_signature(D,n(N,NS),_,SG), commit_template(n(N,NS),S,I).

% Handling additional constraints due to already assembled parts. Unify
% each existing object with one and only one hierarchy node.
1{ unify_node(X,N) : commit_choice(N,C) }1 :- ext_node(X,C).
:- unify_node(X1, N1), unify_node(X2, N2), X1 != X2, N1 = N2.
to_connect(N1,N2) :- to_connect(N1,N2,_,_).     % Project 4->2
:- ext_edge(X1,X2), unify_node(X1,N1), unify_node(X2,N2), not to_connect(N1,N2).

% Edge symmetricity
to_connect(N2,N1) :- to_connect(N1,N2).
ext_edge(X2,X1) :- ext_edge(X1,X2).

% Compute final compatibility score by taking average across all atomic nodes
total_score(TS) :- TS = #sum { X,O : is_likely(O,P,X), use_as(O,P) }.
avg_score(TS/NN) :- total_score(TS), NN = #count { N : atomic_node(N) }, NN != 0.

% Maximize final compatibility score
#maximize { AS : avg_score(AS) }.