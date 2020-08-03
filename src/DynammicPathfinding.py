import time
import os
import glob
import shutil

import json

import math

import itertools

import colorsys

import random as rd

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

import seaborn as sns

import plotly.figure_factory as ff
import plotly.graph_objects as go

# Local Imports
from Multigrid import Multigrid


# DynammicPathfinding Class
class DynammicPathfinding:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, size,
                 sC=0, sV=None, sP=(False, True, True),
                 isValued=True,
                 maxGen=10, printGen=0,
                 tileOutline=False, alpha=1,
                 source=(0, 0, 1, 0), targets={(0, 0, 2, 0)}, numGensDisplayPath=10,
                 borderSet={0, 1, 2, 3, 4, 5, 6}, borderColor='black', borderVal=-1, dispBorder=False,
                 invalidSets=[], invalidColors=[], invalidVals=[], dispInvalid=True,
                 iterationNum=0,
                 aStar=False, untilComplete=True):
        
        self.source = source
        self.targets = targets
        self.targetsList = list(targets)
        
        

        self.numGensDisplayPath = numGensDisplayPath

        self.shortestPath = {'gold': set()}

        self.framesPerRefresh = 10

        self.printPath = False

        self.aStar = aStar
        if self.aStar:
            self.untilComplete = untilComplete

        # Early animation exit param
        self.continueAnimation = True
        # Constant grid parameters
        self.startTime = time.time()

        ## Main Imports ##
        self.dim, self.size = dim, size
        self.sC, self.sV, self.sP = sC, sV, sP
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = sP

        # Generation counters
        self.maxGen, self.printGen = maxGen, printGen

        self.tileOutline, self.alpha, = tileOutline, alpha

        self.borderSet, self.borderColor, self.borderVal, self.dispBorder = borderSet, borderColor, borderVal, dispBorder
        self.invalidSets, self.invalidColors, self.invalidVals, self.dispInvalid = invalidSets, invalidColors, invalidVals, dispInvalid

        # Do not mess with this
        self.tileSize = 10

        # Generate directory names and paths
        self.genDirectoryPaths()
        os.makedirs(self.localPath)

        ## Independent Actions of Constructor ##
        self.invalidSet = set()
        for invalidSet in self.invalidSets:
            self.invalidSet.update(invalidSet)

        # The number of tiles in this grid
        self.numTilesInGrid = (math.factorial(
            self.dim)/(2*(math.factorial(self.dim-2))))*((2*self.size+1)**2)

        # Current tile count
        self.iterInd = 0

        # Create and animate original tiling
        self.currentGrid = Multigrid(self.dim, self.size, shiftVect=self.sV, sC=self.sC, shiftProp=sP,
                                     numTilesInGrid=self.numTilesInGrid,
                                     startTime=self.startTime, rootPath=self.localPath, ptIndex=self.iterInd,
                                     isValued=False,
                                     tileOutline=self.tileOutline, alpha=self.alpha, printGen=self.printGen,
                                     borderSet=self.borderSet, invalidSets=self.invalidSets, borderColor=self.borderColor, invalidColors=self.invalidColors,
                                     borderVal=self.borderVal, invalidVals=self.invalidVals, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid)
        if self.aStar != False:
            tV = list(self.currentGrid.imagToReal(self.currentGrid.tileParamToVertices(
                self.target[0], self.target[2], self.target[1], self.target[3])))
            vList = []
            it = iter(tV)
            for x in it:
                vList.append((x, next(it)))
            self.targetCord = (
                sum([vert[0] for vert in vList])/4, sum([vert[1] for vert in vList])/4)
        self.orderedSourceToTarget = self.orderTargets()
        self.source, self.target = self.orderedSourceToTarget[0]


        self.currentGrid.invalidSet = self.invalidSet

        self.animatingFigure = plt.figure()
        self.ax = plt.axes()
        self.ax.axis('equal')

        self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames,
                                  init_func=self.initPlot(), repeat=False, save_count=self.maxGen)
        self.updateDir()

        print(f"Grid(s) 0-{self.iterInd} Generated succesfully")
        print(f"Grid(s) 0-{self.iterInd-1} Displayed and analyzed succesfully")

        lim = 10
        bound = lim*(self.size+self.dim-1)**1.2
        self.ax.set_xlim(-bound +
                         self.currentGrid.zero[0], bound + self.currentGrid.zero[0])
        self.ax.set_ylim(-bound +
                         self.currentGrid.zero[1], bound + self.currentGrid.zero[1])

        self.saveAndExit()

    def initPlot(self):
        self.ax.cla()

    ########################################
    ## Update Methods For Animation Cycle ##
    ########################################

    def updateDir(self):
        self.anim.save(self.gifPath, writer=PillowWriter(fps=4))

    ########################################
    ########################################
    ########################################
    ########################################
    def updateAnimation(self, i):
        # Axiom 0: All functions and data structures such as the QuadTree and QuadNode classes are implemented correctly
        # This means that all states (and there are many) that this algorithm accumulates on is taken axiomatically as correct
        # even if that is not so (which it in fact is not so).

        # Intent: Generate a single frame containing the tiling and the pathfinding algorithm at index i==iterInd. At creation, updateAnimation should instantiate all neccessary pathfinding variables and should thereafter iterate over the algorithm.

        # Prec 0 : len(invalidSet)==len(invalidCols)

        # Post 0: dijkstra's pathfinding algorithm is iterated and shortest paths are memoized
        # Post 1: if the current source to target path has been explored enough (as defined by user), then a new source and target are chosen and all algorithm variables are reset
        # Post 2: the axis representing the frame containing a visual representation of the tiling and the path finding algorithm (as defined by animA in the requirements) is returned
        # Post 3: currentGrid is set to nextGrid, calculated throughout the method

        # State 0: If the first frame has not been generated, the algorithm is set up
        if self.iterInd == 0:
            self.setupPathfinding()
        # State iterInd: If the first frame has been generated, the algorithm continues to iterate at iterInd
        else:
            # State iterInd.0_a: If the targetInd exceeds the list of source to target problems, the animation is over
            if self.targetInd == len(self.orderedSourceToTarget) - 1:
                self.continueAnimation = False
            # State iterInd.0_b: If the number of generations in which we have found a solution from source to k exceeds
            # numGensDisplayPath, we reset the algorithm to a new source and target
            elif self.numKPrint <= 0:
                # State iterInd.0_b.0: All pointers used for keeping track of the algorithm's indices are reset
                self.targetInd += 1
                self.savedSources.append(self.source)
                self.savedTargets.append(self.target)
                self.source, self.target = self.orderedSourceToTarget[self.targetInd]
                self.kInd = 0
                self.printPath = False
                self.numKPrint = self.numGensDisplayPath
                
                #State iterInd.0_b.1: The shortest path is saved to the solution paths and is then reset
                self.savedPaths.append(self.shortestPath)
                self.shortestPath = set()

                # State iterInd.0_b.2: The minDist and minDistPrev dictionaries are reset
                self.currentGrid.minDistances = {tuple(tileInd): float(
                    'inf') for tileInd in self.currentGrid.allTiles}
                self.currentGrid.minDistPrev = {
                    tuple(tileInd): None for tileInd in self.currentGrid.allTiles}
                self.currentGrid.minDistances[self.source] = 0

                # State iterInd.0_b.3: The univisted set is reset
                self.currentGrid.unVisited = set()
                for tInd in self.currentGrid.allTiles:
                    t = self.currentGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                    if len(t.neighbourhood) in self.borderSet or tuple(tInd) in self.invalidSet:
                        continue
                    self.currentGrid.unVisited.add(tuple(tInd))
                # State iterInd.0_b.4: The source node is visited and a new curent tile is set
                self.currentGrid.unVisited.remove(self.source)
                self.currentGrid.currTInd = self.source
            # State iterInd.0_c: If a solution has been found, we mark that we spent this algorithm cycle looking for a better solution
            elif self.printPath:
                self.numKPrint += -1
                
            # State iterInd.1: using the currentTInd (the next tile to be evaluated), the minimum distances and previous dictionaries are updated
            # at the algorithm iteration iterInd.
            if self.currentGrid.unVisited and self.continueAnimation:

                # State iterInd.1.0: The current tile's information is obtained via its index, and then removed from visited
                currTile = self.currentGrid.multiGrid[self.currentGrid.currTInd[0]
                                                      ][self.currentGrid.currTInd[1]][self.currentGrid.currTInd[2]][self.currentGrid.currTInd[3]]
                self.currentGrid.unVisited.discard(self.currentGrid.currTInd)

                # State iterInd.1.n: The current tile's nth neighbour is evaluated
                for nInd in currTile.neighbourhood:
                    # State iterInd.1.n.0: The nth neighbour's data is retreived
                    n = self.currentGrid.multiGrid[nInd[0]
                                                   ][nInd[1]][nInd[2]][nInd[3]]
                    # State iterInd.1.n.1: Any invalid or border tile is ignored
                    if len(n.neighbourhood) in self.borderSet or tuple(nInd) in self.invalidSet:
                        continue
                    # State iterInd.1.n.2: The nth neighbour's minDistance is calculated and added to the minDistances and previous if it forms a new shortest path
                    currDist = self.currentGrid.minDistances[self.currentGrid.currTInd] + 1
                    # print(self.minDistances[tuple(nInd)])
                    if currDist < self.currentGrid.minDistances[tuple(nInd)]:
                        # State iterInd.1.n.2.0: The minDistance and previous dictionaries are updated
                        self.currentGrid.minDistances[tuple(nInd)] = currDist
                        self.currentGrid.minDistPrev[tuple(
                            nInd)] = self.currentGrid.currTInd
                        # State iterInd.1.n.2.1: If "a" shortest path from the source to the target has been found, printPath is set to true to print the path
                        if not self.printPath and (self.currentGrid.minDistances[self.target] != float('inf')):
                            self.printPath = True

            # State iterInd.2: From the dictionary of minDistances, a list of sets of equally valued tiles is generated
            knownVals = set()
            valToListInd = {}
            listOfValSets = []
            for tInd, dist in self.currentGrid.minDistances.items():
                val = float('inf') if dist == float('inf') else dist
                if val in knownVals:
                    listOfValSets[valToListInd.get(val)].add(tInd)
                else:
                    curSet = set()
                    curSet.add(tInd)
                    listOfValSets.append(curSet)
                    valToListInd[val] = len(listOfValSets) - 1
                    knownVals.add(val)

            # State iterInd.3: A list of sequential colors are generated, one for each set of colors in iterInd.1
            #hsvCols = [(x/len(knownVals), 1, 0.75) for x in range(len(knownVals))]
            #colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvCols))
            #colors = [tuple(color) for color in sns.cubehelix_palette(len(knownVals))]
            distColors = sns.color_palette("husl", len(knownVals))

            # State iterInd.4: The values that we have seen so far are sorted
            sortedKnowns = sorted(list(knownVals))
            # colorSets = {colors[i]:listOfValSets[valToListInd.get(val)] for i, val in enumerate(sortedKnowns)}

            # State iterInd.5: A dictionary of colors and their respective sets of tiles are created
            colorSets = {}
            for i, val in enumerate(sortedKnowns):
                if val == float('inf'):
                    color = 'lightgrey'
                else:
                    color = distColors[i]
                colorSets.update({color: listOfValSets[valToListInd.get(val)]})

            # State iterInd.6: The shortest path and it's color (gold) is added to the above dictionary, if it doesnt exist
            # then this line does nothing
            colorSets.update(self.shortestPath)
            tiles = {tile for path in self.savedPaths for tile in path['gold']}
            colorSets.update({'gold': tiles})
            colorSets.update({'red': {self.target, self.source}})
            colorSets.update({'red': set(self.savedSources)})
            colorSets.update({'red': set(self.savedTargets)})
            # for shortestPath in self.savedPaths:
            #     colorSets.update(shortestPath)
            # colorSets.update({'gold': {self.target}})

            # State iterInd.7: If the shortest path has been found, we iterate back to the source through the previous dictionary
            # and add each tile on the way to a set of tiles in the path.
            if self.printPath:
                if self.iterInd % self.framesPerRefresh == 0:
                    # Remake the path
                    path = set()
                    cur = self.target
                    while cur != self.source:
                        path.add(cur)
                        cur = self.currentGrid.minDistPrev[cur]
                    self.shortestPath = {'gold': path}

            # State iterInd.8: The figure representative of the colorSet we inputted is generated via the native currentGrid method
            # nextSelective, the pre-existing data is cleared
            self.currentGrid.ax.cla()
            nextGrid, axis = self.currentGrid.nextSelective(
                self.currentGrid.currTInd, colorSets, self.source, self.target)

            # State iterInd.9: The next tile index to be evaluated is calculated. For each tile in the tiling, it is given a score based on
            # the minDistances dictionary and the lowest scoring tile is kept track of
            currMin = None
            minVal = float('inf')
            for key, val in self.currentGrid.minDistances.items():

                # State iterInd._a: If we are pathfinding via an A* approximation, then the score of each tile is calculated based on
                # its integer index distance fromt the source found in minDistances but also based on it's estimated distance from the
                # target. Furthermore, a targetBias and sourceBias have been introduced with effects as described below.
                if self.aStar:
                    if val != float('inf') and key in self.currentGrid.unVisited:
                        dist = self.currentGrid.multiGrid[key[0]
                                                          ][key[1]][key[2]][key[3]].dist

                        # targetBias and sourceBias are used to apply biases on searching radially at the target or source respecitively

                        # targetBias, sourceBias = (10,1) is very target biased and may not find a good solution as it will tend to stick to the
                        # minimal resistance path and will not look for better (less-intuative) solutions, this is good if there are little/no invalid tiles

                        # targetBias, sourceBias = (1,20) is very source biased and will find the best solution guaranteed by the time it checks all the
                        # neighbours of the target, this is very similar to Brute forcing with something like Dijkstra's algorithm

                        targetBias, sourceBias = 1, 10
                        score = targetBias*dist + sourceBias*val
                        if score < minVal:
                            currMin, minVal = key, score

                # State iterInd._b: If we are pathfinding via dijkstra's method, the score of each tile is calculated based on
                # its integer index distance from the source.
                else:
                    if val < minVal and val != float('inf') and key in self.currentGrid.unVisited:
                        currMin, minVal = key, val

            # State iterInd.10: All data we have found is calculated and passed down into the next algorithm instance (nextGrid)
            nextGrid.currTInd = currMin
            nextGrid.unVisited = self.currentGrid.unVisited
            nextGrid.minDistances, nextGrid.minDistPrev = self.currentGrid.minDistances, self.currentGrid.minDistPrev

            # if self.aStar and (currMin==None or currMin==float('inf')):
            #     self.continueAnimation = False
            # if currMin == None or len(self.currentGrid.unVisited) < 2 or self.iterInd > self.maxGen-1:
            #     self.continueAnimation = False

            # State iterInd.11: The algorithm state is updated to the next state
            self.currentGrid = nextGrid

            # State iterInd.12: All information regarding the state of the last tile is added to the figure for display
            # Format animation title
            sM = ''
            if self.sP[0]:
                sM = 'zeroes'
            elif self.sP[1]:
                sM = 'random'
            elif self.sP[2]:
                sM = 'halves'
            title = f'dim={self.dim}, size={self.size}, sC={self.sC}, sM={sM}, gen={self.iterInd}'
            self.ax.set_title(title)
            print(f'Grid {self.iterInd} complete', end='\r')

            # State iterInd.13: the index of the algorithm bumps up,  the figure is returned and the algorithm manager saves the frame and moves the
            # algorithm into its next state
            self.iterInd += 1
            # State iterInd.14: the index of the algorithm relative to the current target is bumped up, (this does not help move the algorithm into the next state)
            self.kInd += 1
            return axis

    def setupPathfinding(self):
        # State 0.0: The tiling is mathematically generated and stored
        self.currentGrid.genTilingVerts()
        print('Tile vertices generated')
        self.currentGrid.genTileNeighbourhoods()
        self.currentGrid.genNonDiagTileNeighbourhoods()
        print('Tile neighbourhood generated')
        self.currentGrid.ax.cla()

        # State 0.1: The essential pathfinding dictionaries are instantiated, namely the minDistances, and previous dictionaries
        # The former is instantiated at infinity while the latter is instantiated to nulls
        self.currentGrid.minDistances = {tuple(tileInd): float(
            'inf') for tileInd in self.currentGrid.allTiles}
        self.currentGrid.minDistPrev = {
            tuple(tileInd): None for tileInd in self.currentGrid.allTiles}
        self.currentGrid.minDistances[self.source] = 0

        # State 0.2: The unvisited set is generated such that tiles in the border set and the tiles in the invalid set are not to be visited
        self.currentGrid.unVisited = set()
        # Maybe add marker to source here by changing self.minDistPrev[self.source] to string saying 'source'
        for tInd in self.currentGrid.allTiles:
            t = self.currentGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
            if len(t.neighbourhood) in self.borderSet or tuple(tInd) in self.invalidSet:
                continue
            self.currentGrid.unVisited.add(tuple(tInd))

        # State 0.3: The current index is set to the source given by user input
        self.currentGrid.currTInd = self.source
        # State 0.4: The index of the algorithm bumps up, and the algorithm is moved into pathfinding
        self.iterInd += 1
        # State 0.5 The current index of the pathfinding algorithm in finding the path from the k-1th to kth target
        self.kInd = 0
        self.numKPrint = self.numGensDisplayPath
        # State 0.6: The current index of the target that we are analyzing is stored
        self.targetInd = 0
        # State 0.7: The set of known min source-target paths are saved
        self.savedPaths = []
        self.savedSources = []
        self.savedTargets = []

    #######################
    ## Generator Methods ##
    #######################

    def genFrames(self):
        while (self.iterInd < self.maxGen) and self.continueAnimation:
            self.ax.cla()
            yield self.iterInd
        yield StopIteration

    def genDirectoryPaths(self):
        # rootPath and path data
        filePath = os.path.realpath(__file__)
        self.rootPath = filePath.replace(
            'src/DynammicPathfinding.py', 'outputData/pathMultigridData/')

        self.searchInd = str(rd.randrange(0, 2000000000))
        self.localPath = f"{self.rootPath}searchInd{self.searchInd}/"

        # gifPaths
        searchType = 'aStar' if self.aStar else 'dij'
        self.gifPath = f'{self.localPath}{searchType}Animation.gif'

    #######################
    ## Save All And Exit ##
    #######################

    def saveAndExit(self):
        plt.close('all')
        del self.animatingFigure


    ## MST ##
    def orderTargets(self):
        # State 0.1: The current set of targets is converted to a list of targets containing the source tile
        newTargets = list(self.targets)
        newTargets.append(self.source)
        
        # State 0.2: For each edge in the graph of tile indices (just indices here), we calculate the distance between target/source tiles and add them to the adjacency matrix G
        G = [ [0 for i in range(len(newTargets)) ] for j in range(len(newTargets))]
        for i in range(len(newTargets)):
            for j in range(len(newTargets)):
                if i == j:
                    G[i][j] = 0
                iDist = self.getTargetCord(newTargets[i])
                jDist = self.getTargetCord(newTargets[j])
                dist = math.sqrt( (jDist[1] - iDist[1] )**2 + ( jDist[0] - iDist[0] )**2 )
                G[i][j] = dist
        # State 0.3: The findMST helper method is used to return the set of tuples that define the source and target indices of dijkstra's sub-problems to the MST
        mst = findMST(G)
        
        ## Reformatting don't mind
        sourceTargetPairs = [(newTargets[mid[0]], newTargets[mid[1]]) for mid in mst]
        return sourceTargetPairs
        

    def getTargetCord(self, target):
        tV = list(self.currentGrid.imagToReal(self.currentGrid.tileParamToVertices(
            target[0], target[2], target[1], target[3])))
        vList = []
        it = iter(tV)
        for x in it:
            vList.append((x, next(it)))
        targetCord = (
            sum([vert[0] for vert in vList])/4, sum([vert[1] for vert in vList])/4)
        return targetCord


def readTilingInfo(path):
    with open(path, 'r') as fr:
        tilingInfo = json.load(fr)
        fr.close()
    return tilingInfo


def cleanFileSpace(searchClean=False):
    filePath = os.path.realpath(__file__)
    rootPath = filePath.replace('src/DynammicPathfinding.py', '/outputData/')
    if searchClean:
        files = glob.glob(f'{rootPath}pathMultigridData/*')
        for f in files:
            shutil.rmtree(f)
        time.sleep(1)


def genRandTarget(dim, size, source, targets, allInvalids):
    found = False
    tInd = None

    while (not found):
        r = rd.randrange(0, dim-1)
        s = rd.randrange(r+1, dim)
        a = rd.randrange(-size+2, size-1)
        b = rd.randrange(-size+2, size-1)

        tInd = (r, a, s, b)
        if tInd == source:
            pass
        elif tInd in targets:
            pass
        elif tInd in allInvalids:
            pass
        else:
            found = True
    return (r, a, s, b)

# Sub State fMST


def findMST(G):
    # State fMST.0: the number of vertices (n) and target edges (m) are set based on G
    n = len(G)
    m = n-1
    # State fMST.1: the list of unvisited vertices and the set of vertices in the MST are stored
    pq = []
    mst = set()
    # State fMST.2: the current pointers are set, these include the number of edges in the mst, the current total cost and the current vertex we are on
    numEdges = 0
    totalCost = 0
    curVertex = 0
    visited = set()

    # State fMST.3: while the number of edges to form a tree are still smaller than the cardinality of the current edge set of the mst, the current vertex is evaluated
    while numEdges < m:
        # State fMST.3.i: For each vertex, we check if the edge (on curVertex) exists, if it does, we add it to the pq
        for i in range(n):
            if (G[curVertex][i] != 0):
                pq.append((curVertex, i, G[curVertex][i]))
        # State fMST.3.1: The smallest unvisited edge is taken such that we explore a new node
        minE = minEdge(pq, visited)
        # State fMST.3.2: The new edge is added to the visited and the mst set
        visited.add((minE[0], minE[1]))
        mst.add(minE)
        # State fMST.3.3: The edge is removed from the pq, and the number of edges is increased
        numEdges += 1
        pq.remove(minE)
        curVertex = minE[1]
    print(mst)
    return mst



def minEdge(pq, visited):
    if(len(pq) == 0):
        return None

    minVertex = pq[0]
    for curVertex in pq[1:]:
        if curVertex[2] < minVertex[2] and (curVertex[0], curVertex[1]) not in visited and (curVertex[1], curVertex[0]) not in visited:
            minVertex = curVertex
    return minVertex


###############################
## Local Animation Execution ##
###############################
# Main definition


def main():
    cleanFileSpace(searchClean=True)

    ## Space properties ##
    dim = 5
    size = 7
    # Shift vector properties
    sC = 0
    sV = None
    shiftZeroes, shiftRandom, shiftByHalves = False, True, False
    sP = (shiftZeroes, shiftRandom, shiftByHalves)

    isValued = False

    # Display options
    tileOutline = True
    alpha = 1

    maxGen = 2000
    printGen = 0

    borderSet = {0, 1, 2, 3, 4, 5, 6}
    borderColor = 'black'
    borderVal = 0
    dispBorder = True

    k = 5
    numGensDisplayPath = 30

    invalidSet = set()
    invalidSet2 = set()
    for r in range(dim):
        for s in range(r+1, dim):
            for a in range(-size+1, size-1):
                for b in range(-size+1, size-1):
                    if a in {0, -size+1, -size} or b in {0, size-1, size}:
                        continue
                    invalidSet.add((r, a, s, b))
                    invalidSet.add((r, -a, s, b))
                    invalidSet.add((r, a, s, -b))
                    # invalidSet.add((r, -a, s, -a))

    # for r in range(dim):
    #     for s in range(r+1, dim):
    #         invalidSet.add((r, 0, s, 0))
    #         invalidSet2.add((r, 2, s, 2))

    invalidSet.update([
        # (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 2), (0, 2, 1, 0)
    ])
    # invalidSets.append(invalidSet)
    # invalidSets.append(invalidSet2)
    invalidSets = [invalidSet]

    allInvalids = set()
    for invalidSet in invalidSets:
        for invalid in invalidSet:
            allInvalids.add(invalid)

    invalidColors = ['black']
    #invalidColors = ['black', 'black']
    invalidVals = []
    #invalidVals = [numStates, numStates]
    dispInvalid = True

    source = (0, 0, 1, 0)
    # Place target in second-most outer layer besides boundary
    targets = set()
    for i in range(k):
        newTarget = genRandTarget(
            dim, size, source, targets, allInvalids)
        targets.add(newTarget)

    aStar = False

    currMultigrid = DynammicPathfinding(dim, size, sC=sC, sV=sV, sP=sP,
                                    maxGen=maxGen, printGen=printGen,
                                    isValued=isValued,
                                    tileOutline=tileOutline, alpha=alpha,
                                    source=source, targets=targets, numGensDisplayPath=numGensDisplayPath,
                                    borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
                                    borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid,
                                    iterationNum=0,
                                    aStar=aStar, untilComplete=False)


# Local main call
if __name__ == '__main__':
    main()
