% Make choice for the top node (0), which represents the build target subassembly
1{ node_sa_template(0,S,I) : possible_root_template(S,I) }1.
% Determine possible template options for the root node, from the specified
% build target type and available supertype-subtype relations
possible_root_template(S,I) :- build_target(S), template_option(S,I).
possible_root_template(S2,I) :- build_target(S1), subtype_of(S1,S2), template_option(S2,I).

% Annotating each child of a subassembly node N with its required atomic part
% concept type
node_atomic(n(N,NS),P) :- node_sa_template(N,S,I), req_atomic(S,I,NS,P).

% Meanwhile, a subassembly template may be directly specified for a node
node_sa_template(n(N,NS),S2,I2) :-
    node_sa_template(N,S1,I1), req_template(S1,I1,NS,S2,I2).

% Project to specify subassembly type of subassembly nodes
node_sa(N,S) :- node_sa_template(N,S,I).
node_sa(0,S) :- build_target(S).    % Also account for build target subtype

% Ancestor-descendant relations among nodes
component_node(n(N,NS),N) :- node_atomic(n(N,NS),_).
component_node(n(N,NS),N) :- node_sa(n(N,NS),_).
component_node(N1,N3) :- component_node(N1,N2), component_node(N2,N3).  % Transitive

% An object that is likely to be an instance of a type may fill an atomic node
% with matching type. Each object can fill up to one atomic node, and each
% atomic node can be filled by up to one object.
{ fill_node(O,N,PR) } :- node_atomic(N,P), type_likely(O,P,PR).
:- type_likely(O,_,_), #count { N : fill_node(O,N,_) } > 1.
:- node_atomic(N,_), #count { O : fill_node(O,N,_) } > 1.

% Potential subtype usage transfers to potential supertype usage
type_likely(O,P2,PR) :- type_likely(O,P1,PR), subtype_of(P1,P2).

% Do not allow any idle objects when they can be employed
node_unfilled(N) :- node_atomic(N,_), not fill_node(_,N,_).
:- node_atomic(N,P), node_unfilled(N),
    #count { O : type_likely(O,P,PR), not fill_node(O,_,_) } > 0.

% Identifying all objects selected to fill all descendant atomic nodes of a
% subassembly node
component_obj(O,N) :- node_sa(N,S), component_node(ND,N), fill_node(O,ND,_).

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
must_unify(O,N) :- ext_node(O,P), 1{ node_atomic(_,P) }1, node_atomic(N,P).
% If a neighbor of a uniquely unified object connects with another by known
% contact site, unify (assumption here is that contact sites of each atomic part
% are all uniquely distinguishable)
must_unify(O1,N1) :- must_unify(O2,N2), ext_edge(O1,O2,CP1,CP2),
    to_connect(N1,N2,CP1,CP2).
% If a neighbor of a uniquely unified object has known type, and there exist
% exactly one node with matching type in the neighborhood of the uniquely unified
% node, unify
must_unify(O1,N1) :- must_unify(O2,N2), ext_node(O1,P1), ext_edge(O1,O2),
    1{ node_atomic(N,P1) : to_connect(N,N2,_,_) }1,
    node_atomic(N1,P1), to_connect(N1,N2,_,_).
% If a neighbor of a uniquely unified object has known type, and there exist
% more than one nodes with matching type in the neighborhood of the uniquely
% unified node, *MAY* unify with one of them
may_unify(O1,N1) :- must_unify(O2,N2), ext_node(O1,P1), ext_edge(O1,O2),
    2{ node_atomic(N,P1) : to_connect(N,N2,_,_) },
    node_atomic(N1,P1), to_connect(N1,N2,_,_).
% If one side of existing object connection uniquely unifies to an atomic node
% while the other's type is not specified, the latter *MAY* be unified with one
% of other atomic nodes connected to the unified node
may_unify(O1,N1) :- must_unify(O2,N2), ext_node(O1), ext_edge(O1,O2),
    to_connect(N1,N2,_,_).
% Different objects cannot be forced to unify with the same node
:- must_unify(O1,N1), must_unify(O2,N2), O1 != O2, N1 = N2.
% Observe must_unify relations
:- must_unify(O,N), not fill_node(O,N,_).
% Observe ext_edge relations
:- ext_edge(O1,O2,_,_), must_unify(O1,N1,_), must_unify(O2,N2,_), not to_connect(N1,N2,_,_).
:- ext_edge(O1,O2), must_unify(O1,N1,_), must_unify(O2,N2,_), not to_connect(N1,N2,_,_).

% Supertype-subtype relations for ext_nodes
ext_node(O,P2) :- ext_node(O,P1), subtype_of(P1,P2).

% Edge symmetricity
to_connect(N2,N1,CP2,CP1) :- to_connect(N1,N2,CP1,CP2).
ext_edge(O2,O1,CP2,CP1) :- ext_edge(O1,O2,CP1,CP2).
ext_edge(O2,O1) :- ext_edge(O1,O2).

% Compute final compatibility score by taking average across all atomic nodes,
% offsetting by maximum rule violation severity
total_score(TS) :- TS = #sum { PR,O : fill_node(O,N,PR) }.
avg_score(TS/NN) :-
    total_score(TS), NN = #count { N : node_atomic(N,_) }, NN != 0.
final_score(AS-MV) :- avg_score(AS), MV = #max { S : violation_severity(_,S); 0 }.

#program check(c).
#external query(c).
:- final_score(FS), query(c), FS <= c.