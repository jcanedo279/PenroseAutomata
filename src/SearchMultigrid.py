import time
import os
import glob
import shutil

#import cProfile
#import pstats

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

## Local Imports
from Multigrid import Multigrid


## SearchMultigrid Class
class SearchMultigrid:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, size,
                 sC=0, sV=None, sP=(False,True,True),
                 isValued=True,
                 maxGen=10, printGen=0,
                 tileOutline=False, alpha=1,
                 source=(0,0,1,0), target=(0,0,1,2),
                 borderSet={0,1,2,3,4,5,6}, borderColor='black', borderVal=-1, dispBorder=False,
                 invalidSets=[], invalidColors=[], invalidVals=[], dispInvalid=True,
                 iterationNum=0,
                 aStar=False, untilComplete=True, numDisplayGens=20):
        
        self.source, self.target = source, target
        
        self.shortestPath = {'gold':set()}
        
        self.framesPerRefresh = 10
        
        self.printPath = False
        
        self.aStar = aStar
        if self.aStar:
            self.untilComplete = untilComplete
            self.numDisplayGens = numDisplayGens
        
        
        ## Early animation exit param
        self.continueAnimation = True
        ## Constant grid parameters
        self.startTime = time.time()


        ## Main Imports ##
        self.dim, self.size = dim, size
        self.sC, self.sV, self.sP = sC, sV, sP
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = sP

        ## Generation counters
        self.maxGen, self.printGen =maxGen, printGen

        self.tileOutline, self.alpha, = tileOutline, alpha
        
        self.borderSet, self.borderColor, self.borderVal, self.dispBorder = borderSet, borderColor, borderVal, dispBorder
        self.invalidSets, self.invalidColors, self.invalidVals, self.dispInvalid = invalidSets, invalidColors, invalidVals, dispInvalid
        
        ## Do not mess with this
        self.tileSize = 10
        
        ## Generate directory names and paths
        self.genDirectoryPaths()
        os.makedirs(self.localPath)
        
            
        ## Independent Actions of Constructor ##
        self.invalidSet = set()
        for invalidSet in self.invalidSets:
            self.invalidSet.update(invalidSet)
        
        ## The number of tiles in this grid
        self.numTilesInGrid = (math.factorial(self.dim)/(2*(math.factorial(self.dim-2))))*((2*self.size+1)**2)
        
        ## Current tile count
        self.iterInd = 0

        ## Create and animate original tiling
        self.currentGrid = Multigrid(self.dim, self.size, shiftVect=self.sV, sC=self.sC, shiftProp=sP,
                                     numTilesInGrid=self.numTilesInGrid,
                                     startTime=self.startTime, rootPath=self.localPath, ptIndex=self.iterInd, 
                                     isValued=False,
                                     tileOutline=self.tileOutline, alpha=self.alpha, printGen=self.printGen,
                                     borderSet=self.borderSet, invalidSets=self.invalidSets, borderColor=self.borderColor, invalidColors=self.invalidColors,
                                     borderVal=self.borderVal, invalidVals=self.invalidVals, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid)
        if self.aStar != None:
            tV = list(self.currentGrid.imagToReal(self.currentGrid.tileParamToVertices(self.target[0], self.target[2], self.target[1], self.target[3])))
            vList = []
            it = iter(tV)
            for x in it:
                vList.append((x, next(it)))
            self.targetCord = (sum([vert[0] for vert in vList])/4, sum([vert[1] for vert in vList])/4)
            
        self.currentGrid.invalidSet = self.invalidSet

        self.animatingFigure = plt.figure()
        self.ax = plt.axes()
        self.ax.axis('equal')

        self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False, save_count=self.maxGen)
        self.updateDir()

        print(f"Grid(s) 0-{self.iterInd} Generated succesfully")
        print(f"Grid(s) 0-{self.iterInd-1} Displayed and analyzed succesfully")

        lim = 10
        bound = lim*(self.size+self.dim-1)**1.2
        self.ax.set_xlim(-bound + self.currentGrid.zero[0], bound + self.currentGrid.zero[0])
        self.ax.set_ylim(-bound + self.currentGrid.zero[1], bound + self.currentGrid.zero[1])
        
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
        ## Axiom 0: All functions and data structures such as the QuadTree and QuadNode classes are implemented correctly
        # This means that all states (and there are many) that this algorithm accumulates on is taken axiomatically as correct
        # even if that is not so (which it in fact is not so).
        
        ## Intent: Generate a single frame containing the tiling and the pathfinding algorithm at index i==iterInd. At creation, updateAnimation should
        ## instantiate all neccessary pathfinding variables and should also iterate over the algorithm.
        
        ## Prec 0 : len(invalidSet)==len(invalidCols)
        ## Prec 1 : (source && target) are not in invalidSet and (len(source.neighbourhood) && len(target.neighbourhood)) are not in boundarySet
        
        ## Post 0: dijkstra's or A*'s methods are iterated and minimum distances shortest paths are saved
        ## Post 1: the axis representing the frame containing a visual representation of the tiling and the path finding algorithm is returned
        ## Post 2: currentGrid is set to nextGrid, calculated throughout the method
        
        ## State 0: If the first frame has not been generated, the algorithm is set up
        if self.iterInd==0:
            ## State 0.0: The tiling is mathematically generated and stored
            self.currentGrid.genTilingVerts(target=self.targetCord)
            print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            print('Tile neighbourhood generated')
            self.currentGrid.ax.cla()
            
            ## State 0.1: The essential pathfinding dictionaries are instantiated, namely the minDistances, and previous dictionaries
            ## The former is instantiated at infinity while the latter is instantiated to nulls
            self.currentGrid.minDistances = {tuple(tileInd):float('inf') for tileInd in self.currentGrid.allTiles}
            self.currentGrid.minDistPrev = {tuple(tileInd):None for tileInd in self.currentGrid.allTiles}
            self.currentGrid.minDistances[self.source] = 0
            
            ## State 0.2: The unvisited set is generated such that tiles in the border set and the tiles in the invalid set are not to be visited
            self.currentGrid.unVisited = set()
            #Maybe add marker to source here by changing self.minDistPrev[self.source] to string saying 'source'
            for tInd in self.currentGrid.allTiles:
                t = self.currentGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                if len(t.neighbourhood) in self.borderSet or tuple(tInd) in self.invalidSet:
                    continue
                self.currentGrid.unVisited.add(tuple(tInd))
            
            ## State 0.3: The current index is set to the source given by user input
            self.currentGrid.currTInd = self.source
            ## State 0.4: The index of the algorithm bumps up, and the algorithm is moved into pathfinding
            self.iterInd += 1
        ## State iterInd: If the first frame has been generated, the algorithm continues to iterate at iterInd
        else:
            ## State iterInd.0: using the currentTInd (the next tile to be evaluated), the minimum distances and previous dictionaries are updated
            ## at the algorithm iteration iterInd.
            if self.currentGrid.unVisited and self.continueAnimation:
                
                ## State iterInd.0.0: The current tile's information is obtained via its index, and then removed from visited
                currTile = self.currentGrid.multiGrid[self.currentGrid.currTInd[0]][self.currentGrid.currTInd[1]][self.currentGrid.currTInd[2]][self.currentGrid.currTInd[3]]
                self.currentGrid.unVisited.discard(self.currentGrid.currTInd)
                
                ## State iterInd.0.n: The current tile's nth neighbour is evaluated
                for nInd in currTile.neighbourhood:
                    ## State iterInd.0.n.0: The nth neighbour's data is retreived
                    n = self.currentGrid.multiGrid[nInd[0]][nInd[1]][nInd[2]][nInd[3]]
                    ## State iterInd.0.n.1: Any invalid or border tile is ignored
                    if len(n.neighbourhood) in self.borderSet or tuple(nInd) in self.invalidSet:
                        continue
                    ## State iterInd.0.n.2: The nth neighbour's minDistance is calculated and added to the minDistances and previous if it forms a new shortest path
                    currDist = self.currentGrid.minDistances[self.currentGrid.currTInd] + 1
                    #print(self.minDistances[tuple(nInd)])
                    if currDist < self.currentGrid.minDistances[tuple(nInd)]:
                        ## State iterInd.0.n.2.0: The minDistance and previous dictionaries are updated
                        self.currentGrid.minDistances[tuple(nInd)] = currDist
                        self.currentGrid.minDistPrev[tuple(nInd)] = self.currentGrid.currTInd
                        ## State iterInd.0.n.2.1: If "a" shortest path from the source to the target has been found, printPath is set to true to print the path
                        if not self.printPath and (self.currentGrid.currTInd == self.target or nInd == self.target):
                            self.printPath = True
            
            ## State iterInd.1: From the dictionary of minDistances, a list of sets of equally valued tiles is generated
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
            
            ## State iterInd.2: A list of sequential colors are generated, one for each set of colors in iterInd.1
            #hsvCols = [(x/len(knownVals), 1, 0.75) for x in range(len(knownVals))]
            #colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvCols))
            #colors = [tuple(color) for color in sns.cubehelix_palette(len(knownVals))]
            colors = sns.color_palette("husl", len(knownVals))
            
            ## State iterInd.3: The values that we have seen so far are sorted
            sortedKnowns = sorted(list(knownVals))
            # colorSets = {colors[i]:listOfValSets[valToListInd.get(val)] for i, val in enumerate(sortedKnowns)}
            
            ## State iterInd.4: A dictionary of colors and their respective sets of tiles are created
            colorSets = {}
            for i, val in enumerate(sortedKnowns):
                color = 'lightgrey' if val == float('inf') else colors[i]
                colorSets.update({color:listOfValSets[valToListInd.get(val)]})
                
            ## State iterInd.5: The shortest path and it's color (gold) is added to the above dictionary, if it doesnt exist
            ## then this line does nothing
            colorSets.update(self.shortestPath)
            
            ## State iterInd.6: If the shortest path has been found, we iterate back to the source through the previous dictionary
            ## and add each tile on the way to a set of tiles in the path.
            if self.printPath:
                if self.iterInd%self.framesPerRefresh == 0:
                    ## Remake the path
                    path = set()
                    cur = self.target
                    while cur != self.source:
                        path.add(cur)
                        cur = self.currentGrid.minDistPrev[cur]
                    self.shortestPath = {'gold':path}
                    ## Subtract 1 from display Path
                self.numDisplayGens += -1
            
            ## State iterInd.7: The figure representative of the colorSet we inputted is generated via the native currentGrid method
            ## nextSelective, the pre-existing data is cleared
            self.currentGrid.ax.cla()
            nextGrid, axis = self.currentGrid.nextSelective(self.currentGrid.currTInd, colorSets, self.source, self.target)
                
            ## State iterInd.8: The next tile index to be evaluated is calculated. For each tile in the tiling, it is given a score based on
            ## the minDistances dictionary and the lowest scoring tile is kept track of
            currMin = None
            minVal = float('inf')
            for key, val in self.currentGrid.minDistances.items():
                
                ## State iterInd.8_a: If we are pathfinding via an A* approximation, then the score of each tile is calculated based on
                ## its integer index distance fromt the source found in minDistances but also based on it's estimated distance from the
                ## target. Furthermore, a targetBias and sourceBias have been introduced with effects as described below.
                if self.aStar:
                    if val != float('inf') and key in self.currentGrid.unVisited:
                        dist = self.currentGrid.multiGrid[key[0]][key[1]][key[2]][key[3]].dist
                        
                        ## targetBias and sourceBias are used to apply biases on searching radially at the target or source respecitively
                        
                        ## targetBias, sourceBias = (10,1) is very target biased and may not find a good solution as it will tend to stick to the
                        ## minimal resistance path and will not look for better (less-intuative) solutions, this is good if there are little/no invalid tiles
                        
                        ## targetBias, sourceBias = (1,20) is very source biased and will find the best solution guaranteed by the time it checks all the
                        ## neighbours of the target, this is very similar to Brute forcing with something like Dijkstra's algorithm
                        
                        targetBias, sourceBias = 1, 10
                        score = targetBias*dist + sourceBias*val
                        if score < minVal:
                            currMin, minVal = key, score
                            
                ## State iterInd.8_b: If we are pathfinding via dijkstra's method, the score of each tile is calculated based on 
                ## its integer index distance from the source.
                else:
                    if val < minVal and val != float('inf') and key in self.currentGrid.unVisited:
                        currMin, minVal = key, val
                        
            ## State iterInd.9: All data we have found is calculated and passed down into the next algorithm instance (nextGrid)
            nextGrid.currTInd = currMin
            nextGrid.unVisited = self.currentGrid.unVisited
            nextGrid.minDistances, nextGrid.minDistPrev = self.currentGrid.minDistances, self.currentGrid.minDistPrev
            
            ## STOP THE ANIMATION ##
            ## State iterInd.10: If the animation needs to be stopped, we do that here, the first conditionary ensures that the aStar algorithm has not run
            ## evaluated all tiles, the second conditional insures the aStar algorithm isnt stuck between invalid/boundary tiles, the third conditional ensures
            ## there is still a tile to be visited, that the algorithm hasn't surpassed maxGen, and that there is a next tile to be evaluated.
            if self.aStar and self.numDisplayGens < 2 and not self.untilComplete:
                self.continueAnimation = False
            ## currMin == None insures that the animation quits if there is no way to check the other unVisited tiles
            if self.aStar and (currMin==None or currMin==float('inf')):
                self.continueAnimation = False
            if currMin == None or len(self.currentGrid.unVisited) < 2 or self.iterInd > self.maxGen-1:
                self.continueAnimation = False
            
            ## State iterInd.11: The algorithm state is updated to the next state
            self.currentGrid = nextGrid
                    
            ## State iterInd.12: All information regarding the state of the last tile is added to the figure for display
            ## Format animation title
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
            
            ## State iterInd.13: the index of the algorithm bumps up,  the figure is returned and the algorithm manager saves the frame and moves the
            ## algorithm into its next state
            self.iterInd += 1
            return axis
    
    
    

    #######################
    ## Generator Methods ##
    #######################
    def genFrames(self):
        while (self.iterInd < self.maxGen)  and self.continueAnimation:
            self.ax.cla()
            yield self.iterInd
        yield StopIteration

    def genDirectoryPaths(self):
        ## rootPath and path data
        filePath = os.path.realpath(__file__)
        self.rootPath = filePath.replace('src/SearchMultigrid.py', 'outputData/searchMultigridData/')
        
        self.gridPath = 'fitDisp/'
        self.searchInd = str(rd.randrange(0, 2000000000))
        self.localPath = f"{self.rootPath}searchInd{self.searchInd}/"
            
        ## gifPaths
        searchType = 'aStar' if self.aStar else 'dij'
        self.gifPath = f'{self.localPath}{searchType}Animation.gif'
        

    # def genPngPaths(self, pngName):
    #     pngPath = f'{self.localPath}listInd{self.searchInd[0:3]}{pngName}.png'
    #     return pngPath, pngPath.replace('fitMultigridData', 'unfitMultigridData')
    

    #######################
    ## Save All And Exit ##
    #######################
    def saveAndExit(self):
        plt.close('all')
        del self.animatingFigure




def readTilingInfo(path):
    with open(path, 'r') as fr:
        tilingInfo = json.load(fr)
        fr.close()
    return tilingInfo

def cleanFileSpace(searchClean=False):
    filePath = os.path.realpath(__file__)
    rootPath = filePath.replace('src/SearchMultigrid.py', '/outputData/')
    if searchClean:
        files = glob.glob(f'{rootPath}searchMultigridData/*')
        for f in files:
            shutil.rmtree(f)
        time.sleep(1)
            
                
                
                


###############################
## Local Animation Execution ##
###############################
## Main definition
def main():
    cleanFileSpace(searchClean=True)
    
    ## Space properties ##
    dim = 5
    size = 5
    #Shift vector properties
    sC = 0
    sV = None
    shiftZeroes, shiftRandom, shiftByHalves = False, True, False
    sP = (shiftZeroes, shiftRandom, shiftByHalves)
    
    isValued = False
   
    #Display options
    tileOutline = True
    alpha = 1

    maxGen = 100
    printGen = 0

    borderSet = {0, 1, 2, 3, 4, 5, 6}
    borderColor = 'black'
    borderVal = 0
    dispBorder = True
    


    invalidSet = set()
    invalidSet2 = set()
    for r in range(dim):
        for s in range(r+1, dim):
            for a in range(-size+1, size-1):
                for b in range(-size+1, size-1):
                    if a in {0,-size+1,-size} or b in {0,size-1,size}:
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
    invalidColors = ['black']
    #invalidColors = ['black', 'black']
    invalidVals = []
    #invalidVals = [numStates, numStates]
    dispInvalid = True
    
    source = (0, 0, 1, 0)
    ## Place target in second-most outer layer besides boundary
    target = (dim-3, 4, dim-2, 4)
    
    aStar=True
    untilComplete=False
    numDisplayGens=500
    
    currMultigrid = SearchMultigrid(dim, size, sC=sC, sV=sV, sP=sP,
                                    maxGen=maxGen, printGen=printGen,
                                    isValued=isValued,
                                    tileOutline=tileOutline, alpha=alpha,
                                    source=source, target=target,
                                    borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
                                    borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid,
                                    iterationNum=0,
                                    aStar=aStar, untilComplete=untilComplete, numDisplayGens=numDisplayGens)

## Local main call
if __name__ == '__main__':
    main()