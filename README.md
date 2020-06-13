# If Here For Pretty Pictures
If you are here for the prety pictures and gifs, please go towards the bottom and skip the cs/math

# PenroseAutomata
     A series of algorithms to automate the PT calculated in the PenroseTiler repository
The PenroseAutomata repository comprises fours parts:
- The main python scripts (MultigridTree, Multigrid, MultigridCell)
- The supplementary scripts (QuadTree)
- 2 folders used for caching data ('TrashTrees', 'MultigridTreeData')
- Misc. files (README.md, _pycache_)

# Before Running
There are two important things to know before running these algorithms. Most importantly, make sure that two folders named 'TrashTrees' and 'MultigridTreeData' are both in the local directory (mind the caps). Second, if prompted to download Plotly, you must execute the commands as prompted on the console (if you use python you will thank me later for this).

# How To Run
To familiarize yourself with how these scripts work, run MultigridTree.py locally via main(). Notice that this will create a folder in 'TrashTrees' or 'MultigridTreeData'.
If the tiling automata survives to maxGen generations, it will be palced in 'MultigridTreeData', otherwise it will be placed in 'TrashTrees'.
The Dimmension of the tiling is given by dim, and the size is given by size. Notice the tiling has (dim choose 2)*size*size tiles.
sC of 0 makes a true penrose tiling, while 1/2 makes a generalized tiling.
shiftZeroes, shiftRandom, shiftByHalves should be tempered with at your discretion.
isRadByDim, isRadBySize set streaked and radial initial conditions for the tiling respectively.
numColors is the number of boundaries in the tiling, numStates is the number of total states. Notice that numStates >= numColors
gol is a boolean, setting it to True sets and updates tile values in binary as the game of life does, this can be specified in Multigrid.updateTileValsGOL()

Running these algorithms will generally involve creating MultigridTree objects with different parameters. Feel free to tweak the files to your whims, currently the constructor will generate a gif saved to one of the local sub-directories. Creating a QuadTree object specifically involves creating a 4 dimensional MultigridTree object and transferring its data back and forth between the grid and quadTree objects.

# Brief History
This repository exists in lieu of the PenroseTiler repository. The PenroseTiler repository itself comprises the scripts necessary to run a genetic algorithm used to find the set of valid functions that project from a mother lattice to an arbitrary tiling described be a series of objects. The tilings are evaluated using brute force to calculate the number of white (or untiled) pixels to evaluate the fitness of several functions.

The algorithm is successful in finding two trivial solutions, and one generalized solution. The two trivial solutions are capable of producing generalized crystalline tilings while the generalized solution is much more robust and can handle many more shift vector initial conditions.

This repository was initiated in order to provide a library of functions that act on the more robust of the functions.

# Examples
## Single Frame Tilings
You can think as each of these tilings as its own universe, that is an unfilled universe with no states.
Rather, each state here can be thought of as the type of tiling that comprises the space.
     Let's remember that:
     - if dim is odd: there are (dim-1)/2 types of tiles
     - if dim is even: there are (dim/2)-1 types of tiles
All tilings comprise unit lengths and the vertices are uniformly distributed across the tilings with increasing and decreasing densities, such that each density occurs with frequency of the golden ratio power series.

### The following tilings are generated in high dimmensions with smaller and smaller sizes
![n100.png](Examples/n100.png "n=100 size>5")

![n209Out.png](Examples/n209Out.png "n=209 size>1")

![n401Out.png](Examples/n401Out.png "n=401 size>1 zoomed out")

![n401Mid.png](Examples/n401Mid.png "n=401 size>1 zoomed in slightly")

![n401In.png](Examples/n401In.png "n=401 size>1 zoomed in")

The following images contian over 1.2 million tiles each (not types)

![n501Out.png](Examples/n501Out.png "n=501 size>1")

![n501In.png](Examples/n501In.png "n=501 size>1")


## Cellular Automata
## Convergent merging algorithms:
We can set any initial condition on any tiing (the more symmetric the easier) then apply state transitions for each tile relative to surrounding tile states.
This has the effect of averaging the values in an autonomistic fassion in that each automaton has innate variation in its transition rules that allows for emergent behaviour
![convergentMerge1.gif](Examples/convergentMerge1.gif "convergent merge 1")

![convergentMerge2.gif](Examples/convergentMerge2.gif "convergent merge 2")

### Redundantly mapped states from 7 dimmensions:
![multiMap.gif](Examples/multiMap.gif "multiMap")

### Partially mapped states:
![partialMapEven1.gif](Examples/partialMapEven1.gif "partialMap even 1")

![partialMapEven2.gif](Examples/partialMapEven2.gif "partialMap even 2")

![partialMapOdd1.gif](Examples/partialMapOdd1.gif "partialMap odd 1")

![partialMapOdd2.gif](Examples/partialMapOdd2.gif "partialMap odd 2")

### Deep potential wells as acheieved by certain even dimmensions:
![deepWells1.gif](Examples/deepWells1.gif "deep well 1")

![deepWells2.gif](Examples/deepWells2.gif "deep well 2")

### Combining deep potential wells and redundant mapping:
![deepWellMultiMap1.gif](Examples/deepWellMultiMap1.gif "deep well ultiMap 1")

![deepWellMultiMap2.gif](Examples/deepWellMultiMap2.gif "deep well ultiMap 2")

![deepWellMultiMap3.gif](Examples/deepWellMultiMap3.gif "deep well ultiMap 2")

### Adaptive boundary:
![adaptiveBoundary.gif](Examples/adaptiveBoundary.gif "Adaptive boundary")

### Game of life pattern that diverges:
![divergentGOL.gif](Examples/divergentGOL.gif "gol pattern that diverges")

### Sample oscillators from 10 dimmensions, even dimmensions create deeper state wells:
![oscillatorSampleGOL.gif](Examples/oscillatorSampleGOL.gif "Sample oscillators from 10 dimmensions")