(* ::Package:: *)

(* ============================================================
   Basin of Attraction - Newton's Method in 2D
   + Newton vs. Broyden order of convergence comparison

   \[HorizontalLine]\[HorizontalLine] SYSTEM A (comment out to switch) \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]
   z^3 = 1  in real form
     f1(x,y) = x^3 - 3xy^2 - 1 = 0
     f2(x,y) = 3x^2*y - y^3    = 0

   \[HorizontalLine]\[HorizontalLine] SYSTEM B (uncomment to switch) \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]
   f1(x,y) = x^2 + 4y^2 - 16 = 0
   f2(x,y) = x y^2 - 4       = 0

   Run each block one at a time with Shift+Enter.
   ============================================================ *)


(* \[HorizontalLine]\[HorizontalLine] STEP 1: Define the system and Jacobian \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Comment/uncomment one block at a time.                     *)

(* \[HorizontalLine]\[HorizontalLine] System A: z^3 = 1 \[HorizontalLine]\[HorizontalLine] *)
f1[x_, y_] := x^3 - 3 x y^2 - 1
f2[x_, y_] := 3 x^2 y - y^3

j11[x_, y_] :=  3 x^2 - 3 y^2
j12[x_, y_] := -6 x y
j21[x_, y_] :=  6 x y
j22[x_, y_] :=  3 x^2 - 3 y^2

(* \[HorizontalLine]\[HorizontalLine] System B: ellipse + curve (uncomment to use) \[HorizontalLine]*)
(*f1[x_, y_] := x^2 + 4 y^2 - 16
f2[x_, y_] := x y^2 - 4

j11[x_, y_] :=  2 x
j12[x_, y_] :=  8 y
j21[x_, y_] :=  y^2
j22[x_, y_] :=  2 x y*)


JJ[x_, y_]   := {{j11[x,y], j12[x,y]}, {j21[x,y], j22[x,y]}}
fvec[x_, y_] := {f1[x,y], f2[x,y]}


(* \[HorizontalLine]\[HorizontalLine] STEP 2: Initial guesses for the roots \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Rough guesses from a Desmos plot \[LongDash] Newton will refine them.
   Change these when you change the system. *)

(*   System A guesses (z^3 = 1):*)
rootGuesses = {
  { 1.0,   0.0},
  {-0.5,   0.9},
  {-0.5,  -0.9}
};

(*   System B guesses (ellipse + curve, 4 roots): *)
(*rootGuesses = {
  { 1.9,   1.1},   (* guess 1: first quadrant  *)
  {-1.9,   1.1},   (* guess 2: second quadrant *)
  { 1.9,  -1.1},   (* guess 3: fourth quadrant *)
  {-1.9,  -1.1}    (* guess 4: third quadrant  *)
};*)

nRoots = Length[rootGuesses];


(* \[HorizontalLine]\[HorizontalLine] STEP 3: Settings \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)

tol  = 10^-12;
nmax = 300;


(* \[HorizontalLine]\[HorizontalLine] STEP 4: Linear solver \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Solves  J . delta = rhs  using Mathematica's built-in
   LinearSolve. Returns {0,0} if J is singular or near-singular
   to prevent the Newton iteration from blowing up.           *)

linearSolve[J_, rhs_] :=
  Module[{sol},
    If[Abs[Det[J]] < 10^-12,
      Return[{0.0, 0.0}]
    ];
    sol = LinearSolve[J, rhs];
    If[!VectorQ[sol, NumericQ] || Max[Abs[sol]] > 10^10,
      Return[{0.0, 0.0}]
    ];
    sol
  ]


(* \[HorizontalLine]\[HorizontalLine] STEP 5: Newton's method \[LongDash] records step sizes \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Tracks s_k = ||x^{(k+1)} - x^{(k)}|| at each step.
   This is the standard convergence proxy when the exact
   solution is unknown.
   Returns {stepSizeList, xFinal, yFinal}.                    *)

newtonSteps[x0_, y0_] :=
  Module[{x = N[x0], y = N[y0], k, delta, steps = {}},
    For[k = 1, k <= nmax, k++,
      delta = linearSolve[JJ[x, y], -fvec[x, y]];
      x = x + delta[[1]];
      y = y + delta[[2]];
      AppendTo[steps, Norm[delta]];
      If[Norm[delta] < tol, Break[]]
    ];
    {steps, x, y}
  ]


(* \[HorizontalLine]\[HorizontalLine] STEP 6: Broyden's method \[LongDash] records step sizes \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Uses rank-1 Jacobian update:
     B_new = B + Outer[Times, (df - B.s), s] / (s.s)
   where s = x_new - x_old,  df = f_new - f_old.
   Starts with B = J(x0) as the initial approximation.
   Returns {stepSizeList, xFinal, yFinal}.                    *)

broydenSteps[x0_, y0_] :=
  Module[
    {x = N[x0], y = N[y0],
     B = N @ JJ[x0, y0],
     k, fOld, fNew, s, df, delta, xNew, yNew, steps = {}},
    For[k = 1, k <= nmax, k++,
      fOld  = fvec[x, y];
      delta = linearSolve[B, -fOld];
      xNew  = x + delta[[1]];
      yNew  = y + delta[[2]];
      fNew  = fvec[xNew, yNew];
      s  = {xNew - x, yNew - y};
      df = fNew - fOld;
      B  = B + Outer[Times, (df - B . s), s] / (s . s);
      x  = xNew;
      y  = yNew;
      AppendTo[steps, Norm[s]];          (* fixed: was missing closing ] *)
      If[Norm[s] < tol, Break[]]
    ];
    {steps, x, y}
  ]


(* \[HorizontalLine]\[HorizontalLine] STEP 7: Run both methods from every root guess \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   For each guess, both Newton and Broyden are run.
   The final converged point is printed alongside step count.
   Newton's converged points are stored as refinedRoots and
   used later for basin labeling \[LongDash] replacing the crude guesses.

   whichGuess controls which root the convergence plot uses.  *)

whichGuess = 1;   (* change to 2 or 3 to use a different root *)

refinedRoots = {};

Print["Method      Guess   Steps   Converged to"];
Print[StringRepeat["-", 55]];

Do[
  startX = rootGuesses[[g, 1]];
  startY = rootGuesses[[g, 2]];

  nResult = newtonSteps[startX, startY];
  bResult = broydenSteps[startX, startY];

  nSteps2 = nResult[[1]];   (* step size list for Newton  *)
  bSteps2 = bResult[[1]];   (* step size list for Broyden *)
  nFinal  = {nResult[[2]], nResult[[3]]};
  bFinal  = {bResult[[2]], bResult[[3]]};

  (* collect Newton's refined root for basin labeling *)
  AppendTo[refinedRoots, nFinal];

  Print["Newton      ", g, "       ", Length[nSteps2], "     {",
        NumberForm[nFinal[[1]], {7,4}], ", ",
        NumberForm[nFinal[[2]], {7,4}], "}"];
  Print["Broyden     ", g, "       ", Length[bSteps2], "     {",
        NumberForm[bFinal[[1]], {7,4}], ", ",
        NumberForm[bFinal[[2]], {7,4}], "}"];
  Print[""],

  {g, nRoots}
];

(* pull out the step lists for whichGuess for the convergence plot.
   We add an offset so the starting point is NOT already at the root \[LongDash]
   otherwise Newton converges in 1 step and there are no pairs to plot. *)
convN = newtonSteps[
  rootGuesses[[whichGuess, 1]] + 0.5,
  rootGuesses[[whichGuess, 2]] + 0.4
];
convB = broydenSteps[
  rootGuesses[[whichGuess, 1]] + 0.5,
  rootGuesses[[whichGuess, 2]] + 0.4
];
nSteps = convN[[1]];
bSteps = convB[[1]];

Print["Convergence plot uses guess ", whichGuess,
      " with offset start: {",
      rootGuesses[[whichGuess,1]] + 0.5, ", ",
      rootGuesses[[whichGuess,2]] + 0.4, "}"];


(* \[HorizontalLine]\[HorizontalLine] STEP 8: Order of convergence plot \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Plot log10(s_{k+1}) vs log10(s_k).
   The slope equals the order of convergence:
     slope ~ 2   -> quadratic  (Newton)
     slope ~ 1-2 -> superlinear (Broyden)
   Dashed reference lines for slope 1 and slope 2.            *)

logPairs[steps_] := Module[{pairs},
  pairs = Table[
    {Log10[steps[[k]]], Log10[steps[[k + 1]]]},
    {k, Length[steps] - 1}
  ];
  Select[pairs, (#[[1]] > -13 && #[[2]] > -13) &]
]

nLogPairs = logPairs[nSteps];
bLogPairs = logPairs[bSteps];

Print["Newton log-pairs: ", Length[nLogPairs],
      "   Broyden log-pairs: ", Length[bLogPairs]];

If[Length[nLogPairs] < 2,
  Print["Not enough Newton steps to plot \[LongDash] increase the offset in Step 7."];
  Abort[]
];

xRef = nLogPairs[[1, 1]];
yRef = nLogPairs[[1, 2]];

line1 = {Dashed, Gray,         Line[{{xRef, yRef}, {xRef + 4, yRef + 4}}]};
line2 = {Dashed, Darker[Gray], Line[{{xRef, yRef}, {xRef + 2, yRef + 4}}]};

orderPlot = ListPlot[
  {nLogPairs, bLogPairs},
  Joined      -> True,
  PlotMarkers -> {{\[FilledCircle], 9}, {\[FilledSquare], 9}},
  PlotStyle   -> {{Blue, Thickness[0.004]}, {Red, Thickness[0.004]}},
  PlotLegends -> Placed[
    LineLegend[{Blue, Red}, {"Newton", "Broyden"},
      LegendFunction -> "Frame"],
    {0.25, 0.75}],
  Frame       -> True,
  FrameLabel  -> {
    "log\[Subscript]10 s\[Subscript]k",
    "log\[Subscript]10 s\[Subscript](k+1)"},
  PlotLabel   -> Style[
    "Order of Convergence  (slope \[TildeTilde] order)", Bold, 12],
  GridLines      -> Automatic,
  GridLinesStyle -> Directive[LightGray, Dashed],
  Epilog -> {
    line1, line2,
    Text[Style["slope 1", 10, Gray], {xRef + 3.2, yRef + 2.5}],
    Text[Style["slope 2", 10, Gray], {xRef + 1.5, yRef + 3.3}]
  },
  ImageSize -> 480
]


(* \[HorizontalLine]\[HorizontalLine] STEP 9: Newton for basin \[LongDash] single run, no history \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)

newtonBasin[x0_, y0_] :=
  Module[{x = N[x0], y = N[y0], k = 0, delta},
    While[k < nmax,
      delta = linearSolve[JJ[x, y], -fvec[x, y]];
      x = x + delta[[1]];
      y = y + delta[[2]];
      k = k + 1;
      If[!NumericQ[x] || !NumericQ[y] || Abs[x] > 10^8 || Abs[y] > 10^8,
        Break[]
      ];
      If[Max[Abs[delta]] < tol, Break[]]
    ];
    {x, y, k < nmax && Abs[f1[x,y]] < 10^-5 && Abs[f2[x,y]] < 10^-5}
  ]


(* \[HorizontalLine]\[HorizontalLine] STEP 10: Random sampling \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Adjust the domain to match your system.
   System A (z^3=1):          a=-2, b=2, c=-2, d=2
   System B (ellipse+curve): a=-6, b=6, c=-6, d=6        *)

a = -2.0;  b = 2.0;
c = -2.0;  d = 2.0;
N0 = 100000;

SeedRandom[42];
startPts = Table[
  {RandomReal[{a, b}], RandomReal[{c, d}]},
  {N0}
];

Print["Running Newton on ", N0, " random starting points. Please wait..."];

basinResults = Table[
  newtonBasin[startPts[[i, 1]], startPts[[i, 2]]],
  {i, N0}
];

Print["Done."]


(* \[HorizontalLine]\[HorizontalLine] STEP 11: Assign root labels \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]

   Uses refinedRoots from Step 7 (Newton's converged points)
   rather than the crude initial guesses.                     *)

rootTol = 0.1;

labelList = Table[
  If[basinResults[[i, 3]],
    pt    = {basinResults[[i, 1]], basinResults[[i, 2]]};
    dists = Table[Norm[pt - refinedRoots[[k]]], {k, nRoots}];
    If[Min[dists] < rootTol,
      First @ Ordering[dists, 1],
      0
    ],
    0
  ],
  {i, N0}
];


(* \[HorizontalLine]\[HorizontalLine] STEP 12: Basin of attraction plot \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)

(* palette needs at least as many colors as nRoots *)
palette2D = {Blue, Red, Darker[Green], Orange};

pointGraphics = Table[
  {If[labelList[[i]] == 0, White, palette2D[[ labelList[[i]] ]]],
   PointSize[0.006],
   Point[{startPts[[i, 1]], startPts[[i, 2]]}]},
  {i, N0}
];

(* mark the refined roots as small colored dots *)
guessMarkers = Table[
  {Black,           PointSize[0.02], Point[refinedRoots[[k]]],
   palette2D[[k]], PointSize[0.01], Point[refinedRoots[[k]]]},
  {k, nRoots}
];

basinGraphics = Graphics[
  {pointGraphics, guessMarkers},
  Frame          -> True,
  FrameLabel     -> {"x", "y"},
  PlotRange      -> {{a, b}, {c, d}},
  PlotLabel      -> Style[
    "Basin of Attraction", Bold, 13],
  ImageSize      -> 520,
  AspectRatio    -> 1,
  GridLines      -> Automatic,
  GridLinesStyle -> Directive[LightGray, Dashed]
];

(* contour curves: f1=0 in white, f2=0 in yellow *)
curve1 = ContourPlot[
  f1[x, y] == 0, {x, a, b}, {y, c, d},
  ContourStyle -> {Black, Thickness[0.003]},
  PlotRange    -> {{a, b}, {c, d}}
];

curve2 = ContourPlot[
  f2[x, y] == 0, {x, a, b}, {y, c, d},
  ContourStyle -> {Black, Thickness[0.003]},
  PlotRange    -> {{a, b}, {c, d}}
];

basinPlot2D = Show[
  basinGraphics, curve1, curve2,
  PlotRange -> {{a, b}, {c, d}},
  ImageSize -> 520
]
