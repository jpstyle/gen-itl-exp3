#const n = 6.

%% Rules common to answer set planning
holds(F,T+1) :- time(T), holds(F,T), not -holds(F,T+1), T < n.
-holds(F,T+1) :- time(T), -holds(F,T), not holds(F,T+1), T < n.
1{ occ(A,T) : action(A) }1 :- time(T), T < n.
:- occ(A,T), not possible(A,T).

%% Rules common to assembly domains - Static laws
% Consider each atomic object to fill a spot in the structure (or lack thereof)
% (somewhat) equally for the purpose of planning
atomic(O) :- object(O).
atomic(NO) :- non_object(NO).

% Each atomic object to fill a spot in the structure (or lack thereof) is asssociated
% with the corresponding singleton 'subassembly', indexed with the (non-)object name
subassembly(O) :- atomic(O).
part_of(O,O) :- atomic(O).
holds(sa_active(O),0) :- atomic(O).

% Part connection is a symmetric relation
to_connect(O2,O1,P2,P1) :- to_connect(O1,O2,P1,P2).
holds(connected(O2,O1,P2,P1),T) :- holds(connected(O1,O2,P1,P2),T).

% Mark whether connection between two objects is 'real' (object-object) or
% 'speculated' (at least one non-object involved)
conn_real(O1,O2,P1,P2) :- to_connect(O1,O2,P1,P2), object(O1), object(O2).
conn_spec(O1,O2,P1,P2) :- to_connect(O1,O2,P1,P2), not conn_real(O1,O2,P1,P2).

% Triggers collision-free assembly path feasibility check (path sampling would also
% be triggered if needed); consult oracle via the theory reasoning feature ('physical
% collision [theory]')

% At each time step T, certain connections (yet to be made) are more 'prioritized'
% if they feature real objects at both ends and do not result in a dead-end state.
% Mark if there are no prioritized connections to make---which would enable
% making connections of lesser priority.
holds(conn_prioritized(O1,O2,P1,P2),T) :- time(T),
    conn_real(O1,O2,P1,P2), not holds(connected(O1,O2,P1,P2),T),
    holds(sa_active(S1),T), holds(sa_active(S2),T),
    part_of(O1,S1), part_of(O2,S2), not dead_end(S1,S2).
holds(no_prioritized_pairs,T) :- time(T), not holds(conn_prioritized(_,_,_,_),T).

%% Rules common to assembly domains - Dynamic laws
% Pick up a subassembly on table with an empty manipulator
action(pickUp(M,S)) :- manipulator(M), subassembly(S).
possible(pickUp(M,S),T) :- holds(manip_free(M),T), holds(onTable(S),T).
holds(held(M,S),T+1) :- occ(pickUp(M,S),T).
-holds(manip_free(M),T+1) :- occ(pickUp(M,S),T).
-holds(onTable(S),T+1) :- occ(pickUp(M,S),T).

% Drop a subassembly held with a manipulator onto table
action(drop(M,S)) :- manipulator(M), subassembly(S).
possible(drop(M,S),T) :- holds(held(M,S),T).
-holds(held(M,S),T+1) :- occ(drop(M,S),T).
holds(manip_free(M),T+1) :- occ(drop(M,S),T).
holds(onTable(S),T+1) :- occ(drop(M,S),T).

% Assembling two subassemblies; abstracts aligning & tightening. Different
% preconditions depending on whether the intended connection is 'real' or
% 'speculative'. Also tracks index of the newly joined subassembly instance.
% (Note: This implementation assumes we always assemble from right to left,
% without loss of generality)
action(join(S1,S2,O1,O2,P1,P2)) :-
    subassembly(S1), subassembly(S2), to_connect(O1,O2,P1,P2).
possible(join(S1,S2,O1,O2,P1,P2),T) :-
    part_of(O1,S1), part_of(O2,S2), holds(held(l,S1),T), holds(held(r,S2),T),
    holds(conn_prioritized(O1,O2,P1,P2),T).
possible(join(S1,S2,O1,O2,P1,P2),T) :-
    part_of(O1,S1), part_of(O2,S2), holds(held(l,S1),T), holds(held(r,S2),T),
    conn_spec(O1,O2,P1,P2), holds(no_prioritized_pairs,T).
subassembly(S+1) :- occ(join(S1,S2,O1,O2,P1,P2),T), holds(max_sa_index(S),T).
part_of(O,S+1) :- occ(join(S1,S2,_,_,P1,P2),T), holds(max_sa_index(S),T), part_of(O,S1).
part_of(O,S+1) :- occ(join(S1,S2,_,_,P1,P2),T), holds(max_sa_index(S),T), part_of(O,S2).
holds(connected(O1,O2,P1,P2),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
-holds(sa_active(S1),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
-holds(sa_active(S2),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
-holds(held(l,S1),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
-holds(held(r,S2),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
holds(manip_free(r),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T).
holds(sa_active(S+1),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T), holds(max_sa_index(S),T).
holds(held(l,S+1),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T), holds(max_sa_index(S),T).
holds(max_sa_index(S+1),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T), holds(max_sa_index(S),T).
-holds(max_sa_index(S),T+1) :- occ(join(S1,S2,O1,O2,P1,P2),T), holds(max_sa_index(S),T).
% Connection no longer prioritized right after it is made by a join action;
% note how this disabling effect has to be symmetric, or problems will be made
% unsatisfiable
-holds(conn_prioritized(O1,O2,P1,P2),T+1) :-
    occ(join(S1,S2,O1,O2,P1,P2),T), holds(conn_prioritized(O1,O2,P1,P2),T).
-holds(conn_prioritized(O2,O1,P2,P1),T+1) :-
    occ(join(S1,S2,O1,O2,P1,P2),T), holds(conn_prioritized(O1,O2,P1,P2),T).

% Note: We don't consider 'disassemble' operation here; would need to treat
% subassembly predicate as fluent if we were to do this

%% Goal condition; all objects to be connected (at the respective contact point
%% pairs) have been 'connected' at time step n, either truly or speculatively
goal :- holds(connected(O1,O2,P1,P2),n) : to_connect(O1,O2,P1,P2);
    holds(manip_free(M),n) : manipulator(M).
:- not goal.

%% Problem init; common to all problems
% Time steps and manipulators
time(0..n).
manipulator(l). manipulator(r).

% Initial fluent values
holds(manip_free(l),0). holds(manip_free(r),0).
holds(max_sa_index(-1),0).

% % Toy example
% object(x). object(y). non_object(z).
% obj_type(x,1). obj_type(y,2). obj_type(z,3).
% contact(p12). contact(p21). contact(p13). contact(p31).
% to_connect(x,y,p12,p21). to_connect(x,z,p13,p31).
% holds(onTable(x),0). holds(onTable(y),0). holds(onTable(z),0).
% % dead_end(S1,S2) :- part_of(x,S1), part_of(y,S2), not part_of(z,S1), not part_of(z,S2).
% % dead_end(S1,S2) :- part_of(y,S1), part_of(x,S2), not part_of(z,S1), not part_of(z,S2).