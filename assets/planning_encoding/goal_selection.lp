% Make choice for the top node (0), which represents the build target subassembly
1{ node_sa_template(0,S,I) : template_option(S,I) }1 :- build_target(S).

% Annotating each child of a subassembly node N with its required atomic part
% concept type
node_atomic(n(N,NS),P) :- node_sa_template(N,S,I), req_atomic(S,I,NS,P).

% Meanwhile, a subassembly template may be directly specified for a node
node_sa_template(n(N,NS),S2,I2) :-
    node_sa_template(N,S1,I1), req_template(S1,I1,NS,S2,I2).

% An object that may be an instance of atomic part type P can be considered and
% used as one, but an object cannot be used as more than one atomic part type
% at the same time
{ use_as(O,P) } :- is_likely(O,P,_).
:- is_likely(O,_,_), #count { P : use_as(O,P) } > 1.

% Usage should not exceed demand
part_demand(P,DM) :- atomic(P), DM = #count { N : node_atomic(N,P) }.
part_usage(P,PU) :- atomic(P), PU = #count { O : use_as(O,P) }.
:- part_demand(P,DM), part_usage(P, PU), PU > DM.

% Do not allow any idle objects when they can be employed
unmet_demand(P,DM-PU) :- part_demand(P,DM), part_usage(P, PU).
ununsed_supply(P,US) :- atomic(P),
    US = #count { O : is_likely(O,P,_), not use_as(O,_) }.
:- unmet_demand(P,UD), ununsed_supply(P,US), UD > 0, US > 0.

% For tracking which nodes will have to be connected at which contact points
to_connect(D1,D2,CP1,CP2) :-
    node_sa_template(N,S,I),
    fits_signature(D1,N,NS1,SG1), fits_signature(D2,N,NS2,SG2),
    connection_signature(S,I,NS1,NS2,SG1,SG2,CP1,CP2).
fits_signature(n(N,NS),N,NS,NS) :- node_atomic(n(N,NS),_).
fits_signature(D,N,NS,c(NS,SG)) :-
    fits_signature(D,n(N,NS),_,SG), node_sa_template(n(N,NS),_,_).

% Handling additional constraints due to already assembled parts. Unify
% each existing object with one and only one hierarchy node.
1{ unify_node(X,N) : node_atomic(N,C) }1 :- ext_node(X,C).
:- unify_node(X1,N1), unify_node(X2,N2), X1 != X2, N1 = N2.
:- unify_node(X1,N1), unify_node(X2,N2),
    ext_edge(X1,X2,CP1,CP2), not to_connect(N1,N2,CP1,CP2).

% Edge symmetricity
to_connect(N2,N1,CP2,CP1) :- to_connect(N1,N2,CP1,CP2).
ext_edge(X2,X1,CP2,CP1) :- ext_edge(X1,X2,CP1,CP2).

% Compute final compatibility score by taking average across all atomic nodes
total_score(TS) :- TS = #sum { X,O : is_likely(O,P,X), use_as(O,P) }.
avg_score(TS/NN) :-
    total_score(TS), NN = #count { N : node_atomic(N,_) }, NN != 0.

% Maximize final compatibility score
#maximize { AS : avg_score(AS) }.