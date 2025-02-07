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

% Handling additional constraints due to already assembled parts. Atomic
% part type of existing object may or may not be specified, represented
% by arity of ext_node predicate (ext_node/2 vs. ext_node/1). Similarly,
% contact sites between two joined existing parts may or may not be known
% (ext_edge/4 vs. ext_edge/2).

% If an existing object with known part type has exactly one atomic node with
% the matching type, unify
must_unify(X,N) :- ext_node(X,P), 1{ node_atomic(_,P) }1, node_atomic(N,P).
% If a neighbor of a uniquely unified object connects with another by known
% contact site, unify (assumption here is that contact sites of each atomic part
% are all uniquely distinguishable)
must_unify(X1,N1) :- must_unify(X2,N2), ext_edge(X1,X2,CP1,CP2),
    to_connect(N1,N2,CP1,CP2).
% If a neighbor of a uniquely unified object has known type, and there exist
% exactly one node with matching type in the neighborhood of the uniquely unified
% node, unify
must_unify(X1,N1) :- must_unify(X2,N2), ext_node(X1,P1), ext_edge(X1,X2),
    1{ node_atomic(N,P1) : to_connect(N,N2,_,_) }1,
    node_atomic(N1,P1), to_connect(N1,N2,_,_).
% If a neighbor of a uniquely unified object has known type, and there exist
% more than one nodes with matching type in the neighborhood of the uniquely
% unified node, *MAY* unify with one of them
may_unify(X1,N1) :- must_unify(X2,N2), ext_node(X1,P1), ext_edge(X1,X2),
    2{ node_atomic(N,P1) : to_connect(N,N2,_,_) },
    node_atomic(N1,P1), to_connect(N1,N2,_,_).
% If one side of existing object connection uniquely unifies to an atomic node
% while the other's type is not specified, the latter *MAY* be unified with one
% of other atomic nodes connected to the unified node
may_unify(X1,N1) :- must_unify(X2,N2), ext_node(X1), ext_edge(X1,X2),
    to_connect(N1,N2,_,_).
% Different objects cannot be forced to unify with the same node
:- must_unify(X1,N1), must_unify(X2,N2), X1 != X2, N1 = N2.
% Connections between uniquely unified objects enforce node connections
:- must_unify(X1,N1), must_unify(X2,N2), ext_edge(X1,X2),
    not to_connect(N1,N2,_,_).

% Edge symmetricity
to_connect(N2,N1,CP2,CP1) :- to_connect(N1,N2,CP1,CP2).
ext_edge(X2,X1,CP2,CP1) :- ext_edge(X1,X2,CP1,CP2).
ext_edge(X2,X1) :- ext_edge(X1,X2).

% Compute final compatibility score by taking average across all atomic nodes
total_score(TS) :- TS = #sum { PR,O : is_likely(O,P,PR), use_as(O,P) }.
avg_score(TS/NN) :-
    total_score(TS), NN = #count { N : node_atomic(N,_) }, NN != 0.

#program check(c).
#external query(c).
:- avg_score(AS), query(c), AS <= c.