%% Archived version where low-level manipulator movement actions were
%% considered, such as 'pick-up' and 'drop'. Decided to abstract and
%% only consider join actions since it turned out too costly to include
%% all such low-level actions in such plans. Once a sequence of join
%% actions is specified, low-level actions can be readily interpolated
%% which is what current implementation does.

%% Rules common to answer set planning
#program base.
time(0).
holds(F,0) :- init(F).

#program step(t).
time(t+1).
holds(F,t+1) :- holds(F,t), not -holds(F,t+1).
-holds(F,t+1) :- -holds(F,t), not holds(F,t+1).
1{ occ(A,t) : possible(A,t) }1.

#program check(t).
#external query(t).
:- query(t), not goal(t).

%% Rules common to (our way of encoding) assembly domains - Static laws
#program base.
% Mark whether connection between two objects is 'real' (object-object) or
% 'speculated' (at least one non-object involved)
conn_real(O1,O2) :- to_connect(O1,O2), object(O1), object(O2).
conn_spec(O1,O2) :- to_connect(O1,O2), not conn_real(O1,O2).

% Part connection is a symmetric relation
to_connect(O2,O1) :- to_connect(O1,O2).

% Track every occasion where a 'speculative' connection was made earlier than
% a 'real' one; total count of such occurrences will be minimized
#program step(t).
penalize(t,T) :- time(T), t > T,
    conn_real(O1_1,O2_1), occ(join(_,_,O1_1,O2_1),t),
    conn_spec(O1_2,O2_2), occ(join(_,_,O1_2,O2_2),T).

%% Rules common to assembly domains - Dynamic laws
#program step(t).
% pick_up/2: Pick up a subassembly on table with an empty manipulator\
possible(pick_up(M,S),t) :- holds(manip_free(M),t), holds(on_table(S),t).
holds(held(M,S),t+1) :- occ(pick_up(M,S),t).
-holds(manip_free(M),t+1) :- occ(pick_up(M,S),t).
-holds(on_table(S),t+1) :- occ(pick_up(M,S),t).

% drop/2: Drop a subassembly held with a manipulator onto table\
possible(drop(M,S),t) :- holds(held(M,S),t).
-holds(held(M,S),t+1) :- occ(drop(M,S),t).
holds(manip_free(M),t+1) :- occ(drop(M,S),t).
holds(on_table(S),t+1) :- occ(drop(M,S),t).

% join/4: Assembling two subassemblies at specified object; abstracts
% aligning & tightening.
% (Note: This implementation assumes we always assemble from right to left,
% without loss of generality)
possible(join(S1,S2,O1,O2),t) :-
    S1 != S2, O1 != O2,
    holds(held(l,S1),t), holds(held(r,S2),t),
    holds(part_of(O1,S1),t), holds(part_of(O2,S2),t),
    to_connect(O1,O2), not holds(connected(O1,O2),t).
holds(connected(O1,O2),t+1) :- occ(join(S1,S2,O1,O2),t).
holds(connected(O2,O1),t+1) :- occ(join(S1,S2,O1,O2),t).    % Symmetric
holds(manip_free(r),t+1) :- occ(join(S1,S2,O1,O2),t).
-holds(held(l,S1),t+1) :- occ(join(S1,S2,O1,O2),t).
-holds(held(r,S2),t+1) :- occ(join(S1,S2,O1,O2),t).
holds(held(l,S+1),t+1) :- occ(join(S1,S2,O1,O2),t), holds(max_sa_index(S),t).
holds(max_sa_index(S+1),t+1) :- occ(join(S1,S2,O1,O2),t), holds(max_sa_index(S),t).
-holds(max_sa_index(S),t+1) :- occ(join(S1,S2,O1,O2),t), holds(max_sa_index(S),t).
-holds(part_of(O,S1),t+1) :- occ(join(S1,S2,_,_),t), holds(part_of(O,S1),t).
-holds(part_of(O,S2),t+1) :- occ(join(S1,S2,_,_),t), holds(part_of(O,S2),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,S2,_,_),t), holds(max_sa_index(S),t), holds(part_of(O,S1),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,S2,_,_),t), holds(max_sa_index(S),t), holds(part_of(O,S2),t).

% Note: We don't consider 'disassemble' operation here; would need to treat
% subassembly predicate as fluent if we were to do this

% Minimize number of penalties due to making speculative joins first; it is
% important that this optimization statement is placed here, as part of the
% program fragment step(t)
#minimize { 1,T1,T2 : penalize(T1,T2) }.

%% Goal condition; all object pairs to be connected have been 'connected'
% at time step t, either truly or speculatively
#program check(t).
goal(t) :- holds(connected(O1,O2),t) : to_connect(O1,O2);
    holds(manip_free(M),t) : manipulator(M).        % Freeing hands

%% Problem init; common to all problems
#program base.
% Manipulators
manipulator(l). manipulator(r).

% Initial configs
init(manip_free(l)). init(manip_free(r)).
init(max_sa_index(-1)).

% 'Base penalty' literal for stabilizing optimization solving mode; w/o this,
% clingo will enumerate all models sharing the same cost value
penalize(0,0).