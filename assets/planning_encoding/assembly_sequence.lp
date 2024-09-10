#const n = 5.

%% Problem init; entity types
time(0..n).
manipulator(l). manipulator(r).
bolt(b). subassembly(b).
wheel(w). subassembly(w).
fender(f). subassembly(f).

%% Problem init; initial fluent values
holds(onTable(b), 0). holds(onTable(w), 0). holds(onTable(f), 0).
holds(manipulatorFree(l), 0). holds(manipulatorFree(r), 0).
holds(max_sa_index(-1), 0).

%% Goal condition
goal :-
    subassembly(S), wheel(S1), bolt(S2), fender(S3),
    holds(partOf(S1, S), n), holds(partOf(S2, S), n), holds(partOf(S3, S), n).
:- not goal.

%% Rules common to answer set planning
holds(F, T+1) :- time(T), holds(F, T), not -holds(F, T+1), T < n.
-holds(F, T+1) :- time(T), -holds(F, T), not holds(F, T+1), T < n.
1{ occ(A, T) : action(A) }1 :- time(T), T < n.
:- occ(A, T), not possible(A, T).

%% Rules common to all assembly domains - Static laws
holds(partOf(S1, S3), T) :- holds(partOf(S1, S2), T), holds(partOf(S2, S3), T).
pointOf(f_p(S, I), S) :- subassembly(S), contactPoint(f_p(S, I)).
pointOf(C, S2) :- pointOf(C, S1), holds(partOf(S1, S2), T).

%% Rules common to all assembly domains - Dynamic laws
% Pick up a subassembly on table with an empty manipulator
action(pickUp(M, S)) :- manipulator(M), subassembly(S).
possible(pickUp(M, S), T) :- holds(manipulatorFree(M), T), holds(onTable(S), T).
holds(held(M, S), T+1) :- occ(pickUp(M, S), T).
-holds(manipulatorFree(M), T+1) :- occ(pickUp(M, S), T).
-holds(onTable(S), T+1) :- occ(pickUp(M, S), T).

% Drop a subassembly held with a manipulator onto table
action(drop(M, S)) :- manipulator(M), subassembly(S).
possible(drop(M, S), T) :- holds(held(M, S), T).
-holds(held(M, S), T+1) :- occ(drop(M, S), T).
holds(manipulatorFree(M), T+1) :- occ(drop(M, S), T).
holds(onTable(S), T+1) :- occ(drop(M, S), T).

% Assembling two subassemblies; abstracts aligning & tightening
action(join(S1, S2, C1, C2)) :-
    subassembly(S1), subassembly(S2), contactPoint(C1), contactPoint(C2).
possible(join(S1, S2, C1, C2), T) :-
    holds(held(l, S1), T), holds(held(r, S2), T), compatible(C1, C2), collisionFree(S1, S2, C1, C2).
subassembly(S+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).
-holds(held(l, S1), T+1) :- occ(join(S1, S2, C1, C2), T).
-holds(held(r, S2), T+1) :- occ(join(S1, S2, C1, C2), T).
holds(manipulatorFree(r), T+1) :- occ(join(S1, S2, C1, C2), T).
holds(held(l, S+1), T+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).
holds(partOf(S1, S+1), T+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).
holds(partOf(S2, S+1), T+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).
holds(max_sa_index(S+1), T+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).
-holds(max_sa_index(S), T+1) :- occ(join(S1, S2, C1, C2), T), holds(max_sa_index(S), T).

% Note: We don't consider 'disassemble' operation here; would need to treat
% subassembly predicate as fluent if we were to do this

%% Rules specific to our truck assembly domain; mostly static laws
p_b(f_p(S, 0)) :- bolt(S). contactPoint(C) :- p_b(C).
p_w(f_p(S, 0)) :- wheel(S). contactPoint(C) :- p_w(C).
p_f(f_p(S, 0)) :- fender(S). contactPoint(C) :- p_f(C).
compatible(C1, C2) :- p_b(C1), p_w(C2).
compatible(C1, C2) :- p_w(C1), p_f(C2).

%% Assembly path feasibility check approximation
collisionFree(S1, S2, C1, C2) :-
    subassembly(S1), subassembly(S2), p_b(C1), p_w(C2), pointOf(C1, S1), pointOf(C2, S2).
collisionFree(S1, S2, C1, C2) :-
    subassembly(S1), subassembly(S2), p_w(C1), p_f(C2), pointOf(C1, S1), pointOf(C2, S2).

%#show.
#show occ/2.
%#show X : holds(X, n).