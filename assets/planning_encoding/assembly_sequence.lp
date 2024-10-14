% 'Theory' of physical collision, whose semantics is external to the ASP encoding
% of the planning domain. In effect, consults external oracle (collision-free
% path checker) as needed.
#theory collision_check {
    obj_sym { - : 0, unary };
    &path_free/0 : obj_sym, head
}.

%% Rules common to answer set planning
#program base.
holds(F,0) :- init(F).

#program step(t).
holds(F,t+1) :- holds(F,t), not -holds(F,t+1).
-holds(F,t+1) :- -holds(F,t), not holds(F,t+1).
1{ occ(A,t) : possible(A,t) }1.

#program check(t).
#external query(t).
:- query(t), not goal(t).

%% Rules common to (our way of encoding) assembly domains - Static laws
#program base.
% Consider each atomic object to fill a spot in the structure (or lack thereof)
% (somewhat) equally for the purpose of planning
atomic(O) :- object(O).
atomic(NO) :- non_object(NO).

% Each atomic object to fill a spot in the structure (or lack thereof) is asssociated
% with the corresponding singleton 'subassembly', indexed with the (non-)object name
init(part_of(O,O)) :- atomic(O).

% Mark whether connection between two objects is 'real' (object-object) or
% 'speculated' (at least one non-object involved)
conn_real(O1,O2,P1,P2) :- to_connect(O1,O2,P1,P2), object(O1), object(O2).
conn_spec(O1,O2,P1,P2) :- to_connect(O1,O2,P1,P2), not conn_real(O1,O2,P1,P2).

% Part connection is a symmetric relation
to_connect(O2,O1,P2,P1) :- to_connect(O1,O2,P1,P2).

% Track every occasion where a 'speculative' connection was made earlier than
% a 'real' one; total count of such occurrences will be minimized
#program step(t).
spec_over_real(t,T) :- time(T), t > T,
    conn_real(O1_1,O2_1,P1_1,P2_1), occ(join(_,_,O1_1,O2_1,P1_1,P2_1),t),
    conn_spec(O1_2,O2_2,P1_2,P2_2), occ(join(_,_,O1_2,O2_2,P1_2,P2_2),T).

% Triggers collision-free assembly path feasibility check (path sampling would also
% be triggered if needed)
&path_free{ O : holds(part_of(O,S1),t); -O : holds(part_of(O,S2),t) } :-
    occ(join(S1,S2,O1,O2,P1,P2),t).

%% Rules common to assembly domains - Dynamic laws

#program step(t).
% pickUp/2: Pick up a subassembly on table with an empty manipulator\
possible(pickUp(M,S),t) :- holds(manip_free(M),t), holds(onTable(S),t).
holds(held(M,S),t+1) :- occ(pickUp(M,S),t).
-holds(manip_free(M),t+1) :- occ(pickUp(M,S),t).
-holds(onTable(S),t+1) :- occ(pickUp(M,S),t).

% drop/2: Drop a subassembly held with a manipulator onto table\
possible(drop(M,S),t) :- holds(held(M,S),t).
-holds(held(M,S),t+1) :- occ(drop(M,S),t).
holds(manip_free(M),t+1) :- occ(drop(M,S),t).
holds(onTable(S),t+1) :- occ(drop(M,S),t).

% join/6: Assembling two subassemblies at specified object & contact point pair;
% abstracts aligning & tightening.
% (Note: This implementation assumes we always assemble from right to left,
% without loss of generality)
possible(join(S1,S2,O1,O2,P1,P2),t) :-
    holds(held(l,S1),t), holds(held(r,S2),t),
    holds(part_of(O1,S1),t), holds(part_of(O2,S2),t),
    to_connect(O1,O2,P1,P2), not holds(connected(O1,O2,P1,P2),t).
holds(connected(O1,O2,P1,P2),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t).
holds(connected(O2,O1,P2,P1),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t).    % Symmetric
-holds(held(l,S1),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t).
-holds(held(r,S2),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t).
holds(manip_free(r),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t).
holds(held(l,S+1),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t), holds(max_sa_index(S),t).
holds(max_sa_index(S+1),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t), holds(max_sa_index(S),t).
-holds(max_sa_index(S),t+1) :- occ(join(S1,S2,O1,O2,P1,P2),t), holds(max_sa_index(S),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,S2,_,_,P1,P2),t), holds(max_sa_index(S),t), holds(part_of(O,S1),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,S2,_,_,P1,P2),t), holds(max_sa_index(S),t), holds(part_of(O,S2),t).

% Note: We don't consider 'disassemble' operation here; would need to treat
% subassembly predicate as fluent if we were to do this

%% Goal condition; all objects to be connected (at the respective contact point
%% pairs) have been 'connected' at time step t, either truly or speculatively
#program check(t).
goal(t) :- holds(connected(O1,O2,P1,P2),t) : to_connect(O1,O2,P1,P2);
    holds(manip_free(M),t) : manipulator(M).

% %% Minimize total count of 'speculative-over-real' occasions
% #minimize { 1,T1,T2 : spec_over_real(T1,T2) }.

%% Problem init; common to all problems
#program base.
% Manipulators
manipulator(l). manipulator(r).

% Initial configs
init(manip_free(l)). init(manip_free(r)).
init(max_sa_index(-1)).