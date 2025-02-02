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
% Part connection is a symmetric relation
to_connect(O2,O1) :- to_connect(O1,O2).

% Track every occasion where a subassembly formed in the previous timestep
% doesn't get used in the immediate next timestep (identifiable by index);
% total count of such occurrences will be minimized
#program step(t).
penalize(t) :-
    occ(join(S1,O1,S2,O2),t), holds(max_sa_index(S),t), S1 != S, S2 != S.

%% Rules common to assembly domains - Dynamic laws
#program step(t).
% join/4: Assembling two subassemblies at specified object; abstracts
% aligning & tightening.
% (Note: This implementation assumes we always assemble from right to left,
% without loss of generality)
possible(join(S1,O1,S2,O2),t) :-
    S1 != S2, O1 != O2,
    holds(part_of(O1,S1),t), holds(part_of(O2,S2),t),
    to_connect(O1,O2), not holds(connected(O1,O2),t).
holds(connected(O1,O2),t+1) :- occ(join(S1,O1,S2,O2),t).
holds(connected(O2,O1),t+1) :- occ(join(S1,O1,S2,O2),t).    % Symmetric
holds(max_sa_index(S+1),t+1) :- occ(join(S1,O1,S2,O2),t), holds(max_sa_index(S),t).
-holds(max_sa_index(S),t+1) :- occ(join(S1,O1,S2,O2),t), holds(max_sa_index(S),t).
-holds(part_of(O,S1),t+1) :- occ(join(S1,_,S2,_),t), holds(part_of(O,S1),t).
-holds(part_of(O,S2),t+1) :- occ(join(S1,_,S2,_),t), holds(part_of(O,S2),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,_,S2,_),t), holds(max_sa_index(S),t), holds(part_of(O,S1),t).
holds(part_of(O,S+1),t+1) :-
    occ(join(S1,_,S2,_),t), holds(max_sa_index(S),t), holds(part_of(O,S2),t).

% Note: We don't consider 'disassemble' operation here; would need to treat
% subassembly predicate as fluent if we were to do this

% Minimize number of times occasions where agent does not immediately
% re-use the subassembly just assembled in the previous timestep for the
% next join action, thus increasing the number of primitive actions needed.
% It is important that this optimization statement is placed here, as part
% of the program fragment step(t).
#minimize { 1,T : penalize(T) }.

%% Goal condition; all object pairs to be connected have been 'connected'
% at time step t, either truly or speculatively
#program check(t).
goal(t) :- holds(connected(O1,O2),t) : to_connect(O1,O2).

%% Problem init; common to all problems
#program base.
% Initial configs
init(max_sa_index(-1)).