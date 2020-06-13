# PenroseAutomata
     A series of algorithms to automate the PT calculated in the PenroseTiler repository
The PenroseAutomata repository comprises fours parts:
- The main python scripts (MultigridTree, Multigrid, MultigridCell)
- The supplementary scripts (QuadTree)
- 2 folders used for caching data ('TrashTrees', 'MultigridTreeData')
- Misc. files (README.md, _pycache_)

# Important
**IN ORDER TO RUN**

There are two important things to know before running these algorithms. Most importantly, make sure that two folders named 'TrashTrees' and 'MultigridTreeData' are both in the local directory (mind the caps). Second, if prompted to download Plotly, you must execute the commands as prompted on the console (if you use python you will thank me later for this).

# Running
Running these algorithms will generally involve creating MultigridTree objects with different parameters. Feel free to tweak the files to your whims, currently the constructor will generate a gif saved to one of the local sub-directories. Creating a QuadTree object specifically involves creating a 4 dimensional MultigridTree object and transferring its data back and forth between the grid and quadTree objects.


# Brief History
This repository exists in lieu of the PenroseTiler repository. The PenroseTiler repository itself comprises the scripts necessary to run a genetic algorithm used to find the set of valid functions that project from a mother lattice to an arbitrary tiling described be a series of objects. The tilings are evaluated using brute force to calculate the number of white (or untiled) pixels to evaluate the fitness of several functions.

The algorithm is successful in finding two trivial solutions, and one generalized solution. The two trivial solutions are capable of producing generalized crystalline tilings while the generalized solution is much more robust and can handle many more shift vector initial conditions.

This repository was initiated in order to provide a library of functions that act on the more robust of the functions.

# Examples
## Convergent merging algorithms:
![convergentMerge1.gif](Examples/convergentMerge1.gif "convergent merge 1")

![convergentMerge2.gif](Examples/convergentMerge2.gif "convergent merge 2")

## Redundantly mapped states from 7 dimmensions:
![multiMap.gif](Examples/multiMap.gif "multiMap")

## Partial mapped states:
![partialMapEven1.gif](Examples/partialMapEven1.gif "partialMap even 1")

![partialMapEven2.gif](Examples/partialMapEven2.gif "partialMap even 2")

![partialMapOdd1.gif](Examples/partialMapOdd1.gif "partialMap odd 1")

![partialMapOdd2.gif](Examples/partialMapOdd2.gif "partialMap odd 2")

## Deep potential wells as acheieved by certain even dimmensions:
![deepWells1.gif](Examples/deepWells1.gif "deep well 1")

![deepWells2.gif](Examples/deepWells2.gif "deep well 2")

## Combining deep potential wells and redundant mapping:
![deepWellMultiMap1.gif](Examples/deepWellMultiMap1.gif "deep well ultiMap 1")

![deepWellMultiMap2.gif](Examples/deepWellMultiMap2.gif "deep well ultiMap 2")

![deepWellMultiMap3.gif](Examples/deepWellMultiMap3.gif "deep well ultiMap 2")

## Adaptive boundary:
![adaptiveBoundary.gif](Examples/adaptiveBoundary.gif "Adaptive boundary")

**Game of life pattern that diverges:**

![divergentGOL.gif](Examples/divergentGOL.gif "gol pattern that diverges")

**Sample oscillators from 10 dimmensions, even dimmensions create deeper state wells:**
![oscillatorSampleGOL.gif](Examples/oscillatorSampleGOL.gif "Sample oscillators from 10 dimmensions")