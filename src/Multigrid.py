import os
import time

import math
import itertools
import bisect
import copy

import random as rd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import seaborn as sns

## Local Imports
from MultigridCell import MultigridCell


## Multigrid Class
class Multigrid:
    #################
    ## Init Method ##
    #################
    def __init__(self, dim, size, shiftVect=None, sC=0, shiftProp=(False,True,False),
                numTilesInGrid=None,
                startTime=0, rootPath=None, ptIndex=None,
                numStates=None, colors=None,
                isValued=True, initialValue=(True,False,False,False), valRatio=None,
                boundToCol=None, boundToPC=None,
                isBoundaried=False, bounds=None, boundaryApprox=True,
                gol=False, tileOutline=False, alpha=1, printGen=0,
                borderSet={0,1,2,3,4,5,6}, invalidSets=[], borderColor='black', invalidColors=[],
                borderVal=0, invalidVals=[], dispBorder=False, dispInvalid=True,
                captureStatistics=False):
        
        ## Multigrid object and instantiate its constant parameters
        self.dim, self.size, self.sC = dim, size, sC
        self.shiftProp = shiftProp
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftProp
        if (not shiftVect) and (ptIndex==0):
            self.shiftVect = self.genShiftVector(zeroes=self.shiftZeroes, random=self.shiftRandom, byHalves=self.shiftByHalves)
        else:
            self.shiftVect = shiftVect

        self.numTilesInGrid = numTilesInGrid

        self.startTime = startTime
        ## Figure data
        self.ax = plt.gca()
        self.rootPath = rootPath
        self.ptIndex = ptIndex
        self.gridPath = 'grid' + str(self.ptIndex) + '.png'
        self.pathPng = self.rootPath + self.gridPath
        ## Penrose tile index
        self.ptIndex = ptIndex
        self.printGen = printGen

        ## Number of states and color maps
        self.numStates, self.colors = numStates, colors

        ## Initial value conditions and rule defs
        self.isValued, self.initialValue, self.valRatio = isValued, initialValue, valRatio

        self.boundToCol, self.boundToPC = boundToCol, boundToPC

        ## Boundarying
        self.isBoundaried, self.bounds, self.boundaryApprox = isBoundaried, bounds, boundaryApprox
        self.boundarySet = set(self.bounds)

        ## Game of life settings
        self.gol=gol

        ## Misc parameters
        self.tileSize, self.tileOutline, self.alpha = 10, tileOutline, alpha

        ## adjacencyMatrix of tile types
        self.ttm = self.genttm()

        ## Boundaries and invalid set
        self.borderSet, self.invalidSets, self.borderColor, self.invalidColors = borderSet, invalidSets, borderColor, invalidColors
        self.borderVal, self.invalidVals = borderVal, invalidVals
        self.dispBorder, self.dispInvalid = dispBorder, dispInvalid
        
        self.captureStatistics = captureStatistics
        
        ## Tile statistics
        if self.captureStatistics:
            self.borderSetLen = 0
            self.numStable = 0
            self.numUnstable = 0
            self.numChanged = 0
            self.numInvalid = 0
        
        self.stablePatches = []
        self.stableTiles = []
        self.unstableTiles = []
        self.specialTiles = []

    ###########################
    ## Generate Shift Vector ##
    ###########################
    def genShiftVector(self, zeroes=False, random=False, byHalves=True):
        ## All shifts set to zero
        if zeroes:
            shiftVect = [0.00001 for i in range(self.dim-1)]
            shiftVect.append(self.sC)
            return shiftVect
        ## Take care of 3 dimmensional edge case
        if self.dim==3:
            samplePopulation = list(range(1, 1000, 1))
            sample = rd.sample(samplePopulation, 2)
            samp1, samp2 = sample[0]/2, sample[1]/2
            final = self.sC - samp1 - samp2
            return [samp1, samp2, final]
        popMultiplier = 1
        samplePopulation = list(range(1, popMultiplier*self.dim))
        ## Even dimmensions, something is going wrong here
        if self.dim%2==0:
            lB = math.floor((self.dim/2)-1)
            samp = rd.sample(samplePopulation, lB)
            samp.sort()
            sV = []
            for i in range(lB):
                if byHalves:
                    if random:
                        sV.append(samp[i]/2)
                        sV.append(-samp[i]/2)
                    else:
                        sV.append((i+1)/2)
                        sV.append(-(i+1)/2)
                else:
                    if random:
                        sV.append(samp[i]/(self.dim+5))
                        sV.append(-samp[i]/(self.dim+5))
                    else:
                        sV.append((i+1)/self.dim)
                        sV.append(-(i+1)/self.dim)
            sV.append((1/3)*self.sC)
            sV.append((2/3)*self.sC)
            return sV
        ## Odd dimmensions
        else:
            lB = math.floor((self.dim-1)/2)
            uB = math.ceil((self.dim-1)/2)
            samp = rd.sample(samplePopulation, lB)
            samp.sort()
            sV = []
            for i in range(lB):
                if byHalves:
                    if not random:
                        sV.append((i+1)/2)
                        sV.append(-(i+1)/2)
                    else:
                        sV.append(samp[i]/2)
                        sV.append(-samp[i]/2)
                else:
                    if not random:
                        sV.append((i+1)/self.dim)
                        sV.append(-(i+1)/self.dim)
                    else:
                        sV.append(samp[i]/(self.dim+5))
                        sV.append(-samp[i]/(self.dim+5))
            if lB != uB:
                sV += [0.0]
            sV += [self.sC]
            return sV

    #############################################
    ## Generate Adjacency Matrix Of Tile Types ##
    #############################################
    def genttm(self):
        ## Intent: Find the tile type of all tiles
        if self.dim%2 == 0:
            numTileTypes = int((self.dim/2)-1)
        else:
            numTileTypes = int((self.dim-1)/2)
        self.numTileTypes = numTileTypes
        ## create tile type adjacency matrix
        ttm = [[0 for x in range(self.dim)] for y in range(self.dim)]
        for x in range(self.dim):
            for y in range(1, self.numTileTypes+1):
                if x+y >= self.dim:
                    ttm[x][(x+y)-self.dim] = y-1
                else:
                    ttm[x][x+y] = y-1
                ttm[x][x-y] = y-1
        return ttm

    #####################################
    ## Generate Vertices For All Tiles ##
    #####################################
    def genTilingVerts(self):
        self.multiGrid = [[[[None for b in range(-self.size, self.size+1)] for s in range(self.dim)] for a in range(-self.size, self.size+1)] for r in range(self.dim)]
        self.allTiles = []
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):

                        tileType = self.ttm[r][s]
                        self.allTiles.append((r, a, s, b))
                        
                        p_rs_ab = MultigridCell(self.dim, r, s, a, b, self.shiftVect[r], self.shiftVect[s], tileType)
                        p_rs_ab.setStability(False)

                        ## Initialize vertices
                        vertices = list(self.imagToReal(self.tileParamToVertices(r, s, a, b)))
                        vList = []
                        it = iter(vertices)
                        for x in it:
                            vList.append((x, next(it)))
                        p_rs_ab.setVertices(vList)
                        self.multiGrid[r][a][s][b] = p_rs_ab

    def tileParamToVertices(self, r, s, a, b):
        ## Return a generator object of tile vertices
        normVect = self.genNormVect()
        ## kp is the tile cordinate calculated by using the k function to project the coordinates of some p in G
        if normVect[s-r].imag == 0:
            kp = 1j*(normVect[r]*(b-self.shiftVect[s]) - normVect[s]*(a-self.shiftVect[r])) / 0.00001
        else:
            kp = 1j*(normVect[r]*(b-self.shiftVect[s]) - normVect[s]*(a-self.shiftVect[r])) / normVect[s-r].imag
        k = [0--((kp/i).real+t)//1 for i, t in zip(normVect, self.shiftVect)]
        for k[r], k[s] in [(a, b), (a+1, b), (a+1, b+1), (a, b+1)]:
            yield sum(x*t for t, x in zip(normVect, k))
    
    def imagToReal(self, vertices):
        for vertex in vertices:
            scaledVert = self.tileSize*vertex
            yield from (scaledVert.real, scaledVert.imag)

    def genNormVect(self):
        return [(-1)**((2/self.dim)*i) for i in range(self.dim)]


    ###########################################
    ## Generate Neighbourhoods For All Tiles ##
    ###########################################
    def genTileNeighbourhoods(self):
        ## For each tile, we give it a set of indices {[t.r, t.a, t.s, t.b],...} that can be used to iterate over
        ## to check the tile's local neighbourhood's values
        #
        ## We wish to make a dictionary w/ the keys containing all the vertices in G, where
        ## each key maps to a set of tiles that contain that vertex.
        ##
        ## We do this heavy lifting with sets by converting the list indices into strings
        ## this does afterall take O(n^2*k^2) time
        vertexToTileSet = {}
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        t = self.multiGrid[r][a][s][b]
                        t.nonDiagTileNeighbourhood = []
                        t.diagTileNeighbourhood = []
                        for vertex in  t.vertices:
                            indexAsStr = f"{t.r} {t.a} {t.s} {t.b}"
                            if vertex in vertexToTileSet.keys():
                                neighbourhoodSet = vertexToTileSet.get(vertex)
                                neighbourhoodSet.add(indexAsStr)
                                vertexToTileSet.update({vertex : neighbourhoodSet})
                            else:
                                neighbourhoodSet = set()
                                neighbourhoodSet.add(indexAsStr)
                                vertexToTileSet.update({vertex : neighbourhoodSet})
        setOfNeighbourhoodSets = vertexToTileSet.values()
        listOfNeighbourhoodLists = []
        for neighbourhoodSet in setOfNeighbourhoodSets:
            neighbourhoodList = []
            for tile in neighbourhoodSet:
                tileIndices = tile[:]
                tileIndices = [int(i) for i in tileIndices.split(' ')]
                neighbourhoodList.append(tileIndices)
            listOfNeighbourhoodLists.append(neighbourhoodList)
        ## For each neighbourhood list in list of neighbourhood lists
        visited = set()
        for neighbourhoodList in listOfNeighbourhoodLists:
            neighbourhoodListCopy = neighbourhoodList[:]
            ## For each tile index in this neighbourhood
            for tileIndex in neighbourhoodList:
                tile = self.multiGrid[tileIndex[0]][tileIndex[1]][tileIndex[2]][tileIndex[3]]
                if tuple(tileIndex) in visited:
                    tnh = tile.neighbourhood
                    tempNeighbourhood = neighbourhoodListCopy.copy()
                    tempNeighbourhood = [i for i in tempNeighbourhood if not i in tnh if not i is tileIndex]
                    tnh.extend(tempNeighbourhood)
                    tile.neighbourhood = tnh
                else:
                    neighbourhoodList = []
                    neighbourhoodList = neighbourhoodListCopy.copy()
                    neighbourhoodList = [i for i in neighbourhoodList if not i is tileIndex]
                    tile.neighbourhood = neighbourhoodList
                    visited.add(tuple(tileIndex))
    def genNonDiagTileNeighbourhoods(self):
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        p_rs_ab = self.multiGrid[r][a][s][b]
                        for nInd in p_rs_ab.neighbourhood:
                            n = self.multiGrid[nInd[0]][nInd[1]][nInd[2]][nInd[3]]
                            ## If a tile and its neighbour share two vertices, aka they share an edge
                            if len(set(p_rs_ab.vertices).intersection(set(n.vertices))) == 2:
                                p_rs_ab.nonDiagTileNeighbourhood = p_rs_ab.nonDiagTileNeighbourhood + [nInd]
                            else:
                                p_rs_ab.diagTileNeighbourhood = p_rs_ab.diagTileNeighbourhood + [nInd]
        
    
    #############################
    ## Generate Initial Tiling ##
    #############################
    def genTiling(self):
        ## Called once at start to generate tiling
        #
        ## Create multigrid instance of cell objects
        #if self.numTileTypes>len(self.colors):
            #colors = sns.cubehelix_palette(self.numTileTypes, dark=0.1, light=0.9)
        self.values = []
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        p_rs_ab = self.multiGrid[r][a][s][b]
                        vertices = p_rs_ab.vertices
                        tileType = p_rs_ab.tileType
                        p_rs_ab_ind = (r, a, s, b)
                        
                        if self.captureStatistics and len(p_rs_ab.neighbourhood) in self.borderSet:
                            self.borderSetLen += 1

                        ## Set the origin of the tiling, this ensures a properly centered animation
                        if(r==0 and s==1 and a==0 and b==0):
                            self.zero = [vertices[0][0], vertices[0][1]]

                        ## Invalid value
                        if p_rs_ab_ind in self.invalidSet:
                            for i, invalidSet in enumerate(self.invalidSets):
                                if p_rs_ab_ind in invalidSet:
                                    p_rs_ab.setVal(self.invalidVals[i])
                                    p_rs_ab.setColor(self.invalidColors[i])
                            if self.dispInvalid:
                                self.specialTiles.append(p_rs_ab_ind)
                            if self.captureStatistics:
                                self.numInvalid += 1
                        ## Border Value
                        elif len(p_rs_ab.neighbourhood) in self.borderSet:
                            p_rs_ab.setVal(self.borderVal)
                            p_rs_ab.setColor(self.borderColor)
                            if self.dispBorder:
                                self.specialTiles.append(p_rs_ab_ind)
                            if self.captureStatistics:  
                                self.numInvalid += 1
                        ## Game Of Life
                        elif self.gol:
                            rand=rd.randrange(0, 3)
                            if rand==0:
                                p_rs_ab.setColor('black')
                                p_rs_ab.setVal(1)
                                self.values.append(1)
                            else:
                                p_rs_ab.setColor('white')
                                p_rs_ab.setVal(0)
                                self.values.append(0)
                        ## Randomly Valued
                        elif self.initialValue[0]:
                            val = rd.randrange(1, self.numStates+1)
                            tileColor = ''
                            if val in self.bounds:
                                tileColor = self.boundToCol.get(val)
                            else:
                                tileColor = self.boundToCol.get(self.valToBound(val))
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setVal(val)
                            self.values.append(val)
                        ## Dimmensionally Valued
                        elif self.initialValue[1]:
                            if r==0:
                                val=rd.randrange(0, self.bounds[r]+1)
                            else:
                                val=rd.randrange(self.bounds[r-1], self.bounds[r]+1)
                            tileColor = self.boundToCol.get(self.bounds[r])
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setVal(val)
                            self.values.append(val)
                        ## Radially Valued
                        elif self.initialValue[2]:
                            if abs(a)==0:
                                val=rd.randrange(self.bounds[abs(a)], self.bounds[abs(a)+1]+1)
                            else:
                                val=rd.randrange(self.bounds[abs(a)-1], self.bounds[abs(a)]+1)
                            tileColor = self.boundToCol.get(self.bounds[abs(a)])
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setVal(val)
                            self.values.append(val)
                        ## Tile Type Valued
                        elif self.initialValue[3]:
                            tileType = p_rs_ab.tileType
                            sampleDef = 1000
                            val=rd.randrange(tileType, (tileType+1)*sampleDef)/sampleDef
                            ## In order to avoid 0 and 1 being in same bound, add 1.
                            p_rs_ab.setColor(self.boundToCol.get(tileType+1))
                            p_rs_ab.setVal(val)
                            self.values.append(val)
                        ## Why do I need to do this?
                        self.multiGrid[r][a][s][b] = p_rs_ab
        if self.captureStatistics:
            self.numTiles = len(self.values)
            self.valAvg = sum(self.values)/self.numTiles
            valStdDev = 0
            for value in self.values:
                valStdDev += (value-self.valAvg)**2
            self.valStdDev = math.sqrt(valStdDev/self.numTiles)

    ########################
    ## Update Grid States ##
    ########################
    def genNextValuedGridState(self, animating=False, boundaried=False):
        nextGrid = Multigrid(self.dim, self.size, shiftVect=self.shiftVect, sC=self.sC, shiftProp=self.shiftProp,
                            numTilesInGrid=self.numTilesInGrid,
                            startTime=self.startTime, rootPath=self.rootPath, ptIndex=self.ptIndex+1,
                            numStates=self.numStates, colors=self.colors,
                            isValued=self.isValued, initialValue=self.initialValue, valRatio=self.valRatio,
                            boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                            isBoundaried=self.isBoundaried, bounds=self.bounds, boundaryApprox=self.boundaryApprox,
                            gol=self.gol, tileOutline=self.tileOutline, alpha=self.alpha, printGen=self.printGen,
                            borderSet=self.borderSet, invalidSets=self.invalidSets, borderColor=self.borderColor, invalidColors=self.invalidColors,
                            borderVal=self.borderVal, invalidVals=self.invalidVals, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid,
                            captureStatistics=self.captureStatistics)
        nextGrid.multiGrid, nextGrid.zero = copy.deepcopy(self.multiGrid), self.zero
        nextGrid.invalidSet = self.invalidSet
        
        self.values = []
        self.stableTiles = []
        
        if self.captureStatistics:
            nextGrid.numTiles = self.numTiles
            self.colValDict = {}

        ## Iterate claiscally over each tile in grid
        if self.ptIndex == 0 or (not self.isBoundaried) or self.gol:
            for r in range(self.dim):
                ## We do not want to check the outermost layer as they are non playable
                for a in range(-self.size, self.size+1):
                    for s in range(r+1, self.dim):
                        for b in range(-self.size, self.size+1):
                            oldTile = self.multiGrid[r][a][s][b]
                            newTile = nextGrid.multiGrid[r][a][s][b]
                            newTile.neighbourhood = oldTile.neighbourhood
                            self.genNextTile(oldTile, newTile)
                            if self.captureStatistics and oldTile.color != newTile.color:
                                self.numChanged += 1 
        ## Iterate over the unstable tiles
        else:
            prevUnstableTiles = self.unstableTiles
            self.unstableTiles = []
            for tInd in prevUnstableTiles:
                oldTile = self.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                newTile = nextGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                newTile.neighbour = oldTile.neighbourhood
                self.genNextTile(oldTile, newTile)
                if self.captureStatistics and oldTile.color != newTile.color:
                    self.numChanged += 1 
        self.unstableTiles = list(set(self.unstableTiles))
        ## Capture Stats related to values
        if self.captureStatistics:
            nextGrid.numTiles = len(self.values)
            nextGrid.valAvg = sum(self.values)/self.numTiles
            nextGrid.valStdDev = math.sqrt(sum([(val-nextGrid.valAvg)**2 for val in self.values])/self.numTiles)
        if not self.gol:
            nextGrid.unstableTiles = self.unstableTiles
            nextGrid.specialTiles = self.specialTiles
            ##########################################
            ## This code creates a patch that does not need to be checked next round
            #self.addToStablePatch(self.stableTiles)
            ##########################################
        nextGrid.stablePatches = self.stablePatches
        if boundaried:
            nextGrid.genBoundaryList()
            if self.ptIndex >= self.printGen:
                nextGrid.displayBoundaries()
        else:
            if self.ptIndex >= self.printGen:
                nextGrid.displayTiling()
        ## Capture Stats related to stability
        if self.captureStatistics:
            self.percentStable = self.numStable/self.numTilesInGrid
            self.percentUnstable = self.numUnstable/self.numTilesInGrid
            self.totalPercent = (self.numStable + self.numUnstable)/self.numTilesInGrid
            self.normPercentStable = self.numStable/(self.numStable+self.numUnstable)
            self.normPercentUnstable = self.numUnstable/(self.numStable+self.numUnstable)
        if animating:
            return nextGrid, self.ax
        return nextGrid     

    def genNextTile(self, oldTile, newTile):
        if (oldTile.r, oldTile.a, oldTile.s, oldTile.b) in self.invalidSet:
            for i, invalidSet in enumerate(self.invalidSets):
                if (oldTile.r, oldTile.a, oldTile.s, oldTile.b) in invalidSet:
                    newTile.setVal(self.invalidVals[i])
                    newTile.setColor(self.invalidColors[i])
                    return
        elif len(oldTile.neighbourhood) in self.borderSet:
            newTile.setVal(self.borderVal)
            newTile.setColor(self.borderColor)
            return
        elif self.gol:
            self.updateTileValsGOL(oldTile, newTile)
            self.values.append(oldTile.val) 
        else:
            newTile = self.updateTileVals(oldTile, newTile)
            self.values.append(oldTile.val)
        if type(oldTile.color) is list:
            col = tuple(oldTile.color)
        col = tuple(oldTile.color) if type(oldTile.color) is list else oldTile.color
        if self.captureStatistics:
            ##### ADD UNSTABLE AND NON BOUDNARIED TILES HERE TO HAVE THEM SHOW UP IN COLOR COMP
            if col in self.colValDict:
                self.colValDict[col] += 1
            else:
                self.colValDict[col] = 1

    ########################
    ## Tile Value Updates ##
    ########################
    ## For the merge algorithm
    def updateTileVals(self, oldTile, newTile):
        oldTileIndex = (oldTile.r, oldTile.a, oldTile.s, oldTile.b)
        oldTileneighbourValTotal = 0
        oldTileisStable = True
        currBound = self.valToBound(oldTile.val)
        for neighbour in oldTile.neighbourhood:
            n = self.multiGrid[neighbour[0]][neighbour[1]][neighbour[2]][neighbour[3]]
            oldTileneighbourValTotal += n.val
            nBound = self.valToBound(n.val)
            ## Check if neighbours are invalid, a border or unstable
            if (tuple(neighbour) in self.invalidSet) or (len(n.neighbourhood) in self.borderSet) or (nBound!=currBound):
                if self.captureStatistics:
                    self.numUnstable += 1
                oldTileisStable = False
                oldTile.setStability(False)
                self.unstableTiles.append(oldTileIndex)
                break
        if oldTileisStable:
            if self.captureStatistics:
                self.numStable += 1
            oldTile.setStability(True)
            oldTileIndex = (oldTile.r, oldTile.a, oldTile.s, oldTile.b)
            self.stableTiles.append(oldTileIndex)
        oldTileneighbourValAvg = oldTileneighbourValTotal/len(oldTile.neighbourhood)
        bound = self.valToBound(oldTileneighbourValAvg)
        pc = self.boundToPC.get(bound)
        
        
        # This makes it the merge method
        newTileVal = oldTile.val + (oldTileneighbourValAvg - oldTile.val)*pc
        #maxUpperChange = ((rd.randrange(0, 1000)*(1/1000)))*(self.numStates - newTileVal)
        #newTileVal += maxUpperChange
        #newTileVal = self.valRatio*oldTile.val + (1-self.valRatio)*oldTileneighbourValAvg
        newTileColor = self.boundToCol.get(self.valToBound(newTileVal))

        ## If the tile state has changed, all previously like-stated stable tiles are set to unstable
        if not self.boundaryApprox:
            if newTileColor != oldTile.color:
                for neighbour in oldTile.neighbourhood:
                    if self.multiGrid[neighbour[0]][neighbour[1]][neighbour[2]][neighbour[3]].val != -1:
                        self.unstableTiles.append(tuple(neighbour))
        newTile.setVal(newTileVal)
        newTile.setColor(newTileColor)
    ## For the GOL algorithm
    def updateTileValsGOL(self, oldTile, newTile):
        oldTileneighbourValTotal = 0
        for neighbour in oldTile.neighbourhood:
            n = self.multiGrid[neighbour[0]][neighbour[1]][neighbour[2]][neighbour[3]]
            oldTileneighbourValTotal += n.val
        oldTileIndex = (oldTile.r, oldTile.a, oldTile.s, oldTile.b)
        if (oldTileneighbourValTotal+oldTile.val==0):
            self.numStable += 1
            oldTile.setStability(True)
            self.stableTiles.append(oldTileIndex)
        else:
            oldTile.setStability(False)
            self.unstableTiles.append(oldTileIndex)
        if (oldTileneighbourValTotal+oldTile.val > 3) and (oldTileneighbourValTotal+oldTile.val  < 5):
            newTile.setVal(1)
            newTile.setColor('black')
        else:
            newTile.setVal(0)
            newTile.setColor('white')
    def addToStablePatch(self, stableTiles):
        for stableTileInd in stableTiles:
            stableTile = self.multiGrid[stableTileInd[0]][stableTileInd[1]][stableTileInd[2]][stableTileInd[3]]
            color = self.boundToCol.get(self.valToBound(stableTile.val))
            if stableTileInd in self.invalidSet or len(stableTile.neighbourhood) in self.borderSet:
                if self.dispInvalid:
                    for i, invalidSet in enumerate(self.invalidSets):
                        if stableTileInd in invalidSet:
                            color = self.invalidColors[i]
                else:
                    return
            elif len(stableTile.neighbourhood) in self.borderSet:
                if self.dispBorder:
                    color = self.borderColor
                else:
                    return
            path = self.genPath(stableTile.vertices)
            if self.tileOutline:
                patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha)
            else:
                patch = mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
            self.stablePatches.append(patch)
    ## Update helper methods
    def genPath(self, vertices):
        ## Intent, create a closped polygon path object
        Path = mpath.Path
        path_data = [
            (Path.MOVETO, vertices[0]),
            (Path.LINETO, vertices[1]),
            (Path.LINETO, vertices[2]),
            (Path.LINETO, vertices[3]),
            (Path.CLOSEPOLY, vertices[0]),
            ]
        codes, verts = zip(*path_data)
        return mpath.Path(verts, codes)
    def setValStats(self, numTiles, valAvg, valStdDev):
        self.numTiles = numTiles
        self.valAvg = valAvg
        self.valStdDev = valStdDev 
    def valToBound(self, val):
        ## Search sorted is the perfect function for this application and does the job in O(log(n)) tc
        ## Search sorted returns the index of the first number in self.bounds that is greater then or equal to val
        return self.bounds[np.searchsorted(self.bounds, val)]     


    #########################
    ## Generate Boundaries ##
    #########################
    def genBoundaryList(self):
        ## A list that contains lists where each contained list contains a collection of unstable tile indexes representing the boundaries on an object
        listOfBoundaryLists = []
        ## A dictionary that maps from a tile index to the boundarylist index it belongs to in listOfBoundaryLists
        tileIndToBoundaryListInd = dict()

        for tInd in self.unstableTiles:
            t = self.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
            tBound = self.valToBound(t.val)

            ## First we create a list of neighbours that are in the same state
            boundedNeighbours = []
            for neighbour in t.neighbourhood:
                n = self.multiGrid[neighbour[0]][neighbour[1]][neighbour[2]][neighbour[3]]
                if (self.valToBound(n.val) == tBound) and (not n.isStable):
                    boundedNeighbours.append(tuple(neighbour))
                        
            ## If there are no neighbours of this state, we create a new boundary object
            if len(boundedNeighbours):
                if (not t.isStable) or (tInd in self.specialTiles) or (len(t.neighbourhood) in self.borderSet):
                    listOfBoundaryLists.append([tInd])
                    tileIndToBoundaryListInd[tInd] = len(listOfBoundaryLists)-1
            ## If there are same-stated neighbours
            else:
                newObjectList = [tInd]
                ## We find the union of all their boundaries
                for neighbourInd in boundedNeighbours:
                    neighbourObjInd = tileIndToBoundaryListInd.get(neighbourInd)
                    if neighbourObjInd != None:
                        objList = listOfBoundaryLists[neighbourObjInd]
                        newObjectList.extend([obj for obj in objList])
                        ## We discard the original tuple set that is not independent but rather part of a hole
                        listOfBoundaryLists[neighbourObjInd] = []
                listOfBoundaryLists.append(newObjectList)

                ## And map their indexes to the newObjectList union list
                for neighbourInd in newObjectList:
                    tileIndToBoundaryListInd[neighbourInd] = len(listOfBoundaryLists)-1
        self.listOfBoundaryLists = [boundry for boundry in listOfBoundaryLists if boundry != []]
        self.numBoundaries = len(self.listOfBoundaryLists)
        
    ##########################
    ## Grid Display Methods ##
    ##########################
    def createPatch(self, inputC):
        r, a, s, b = inputC
        t = self.multiGrid[r][a][s][b]
        vertices = t.vertices
        path = self.genPath(vertices)
        #color = self.colors[self.ttm[r][s]]
        ## Check if the tile is invalid
        if (r, a, s, b) in self.invalidSet:
            if self.dispInvalid:
                for i, invalidSet in enumerate(self.invalidSets):
                    if (r, a, s, b) in invalidSet:
                        color = self.invalidColors[i]
                        break                            
        ## Check if the tile is in the outer boundaries
        elif len(t.neighbourhood) in self.borderSet:
            if self.dispBorder:
                color = self.borderColor
            # Adaptive Background
            #################################################################
            adaptiveBoundary = False
            if adaptiveBoundary:
                if self.ptIndex > 0:
                    bound = self.valToBound(self.valAvg)
                    ### THIS BECOMES EXPENSIVE
                    # boundInd = self.bounds.index(bound)
                    boundInd = np.searchsorted(self.bounds, bound)
                    if boundInd == 0:
                        val = rd.randrange(0, bound+1)
                    else:
                        val = rd.randrange(self.bounds[boundInd-1], bound+1)
                    color = self.boundToCol.get(bound)
                    t.setColor(color)
                    t.setVal(val)
            ################################################################
        elif self.gol:
            if t.val==0:
                color = 'white'
            else:
                color = 'black'
        else:
            color = t.color
        # Origin
        # End
        # Playable Space

        ## Create patch and continue
        if self.tileOutline:
            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha)
        else:
            patch = mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
        return patch
    
    ## Display all tiles
    def displayTiling(self, animating=True):
        ## Axiom0: All functions and data structures such as the QuadTree and QuadNode classes are implemented correctly
        # This means that all states (and there are many) that this algorithm accumulates on is taken axiomatically as correct
        # even if that is not so (which it in fact is not so).
        ## Axiom1: If multip is 'multithread', there are at least six available threads
        ## Axiom2: If multip is 'multiprocess', there are enough available cpus such that os.cpu_count()-1 is greater than or equal to one
        
        ## Intent: the intent of displayTiling() is to iterate over all tiles in the tiling and add the proper patch corresponding to a tiling to the figure.
        # Furthermore, we can do this in one of three ways, by creating a pool of threads, by creating a pool of processes, and by traditionally iterating linearly
        # over a loop with a single process and a single thread. Because of Python's GIL (Global Interpreter Lock), we are unable to take advantage of multithreading
        # in any way that I could find, though this would work in other languages.
        
        ## Prec0: animating is an input parameter, either True or False
        ## Prec1: createPatch is a function that takes in a tuple of four integers (r,a,s,b) st: 0<=r<s<self.dim and -self.size<=a<=b<=self.size
        ## Prec2: createPatch is implemented properly (being outside the scope of this project) and returns a patch that can be easily plotted in the axis self.ax
        
        ## Post0: self.patches is a list containing all the patches in the tiling, where each patch is a plottable object comprising vertices, colors, opacities, etc..
        ## Post1: self.ax contains the patches in self.patches
        ## Post2: if the PA is not automated, we save the tiling frames in a folder containing all the tiling frames in the PA animation
        ## Post3: If the tiling is animated and is the original tiling, self.ax is manually returned
        
        ## State0: multip in {'multithreading', 'multiprocessing', allOtherInputs} and the figure is handled by an auxiliary helper method
        self.setFigExtras()
        multip = False
        ## State1: we conditionally map {'multithreading', 'multiprocessing', allOtherInputs} -> {State1.0.0, 1.1.0, 1.2.0}
        if multip == 'multithread':
            ## State1.0.0: All necessary multithreading modules are imported
            from multiprocessing.dummy import Pool as ThreadPool
            ## State1.0.1: For each of the tiles, the input parameter to the createPatch function is added to the list
            inputs = []
            for r in range(self.dim):
                for a in range(-self.size, self.size+1):
                    for s in range(r+1, self.dim):
                        for b in range(-self.size, self.size+1):
                            inputC = (r, a, s, b)
                            inputs.append(inputC)
            ## State 1.0.2: A thread pool of six thread workers is created
            pool = ThreadPool(6)
            ## Stable 1.0.3: The thread pool maps the createPatch function onto all the input parameters
            patches = pool.imap(self.createPatch, inputs)
            pool.close()
            pool.join()
            ## Stable1.0.4: All the patches are added to the axis self.ax for later display
            self.patches = []
            for patch in patches:
                self.ax.add_patch(patch)
                self.patches.append(patch)
        elif multip == 'multiprocess':
            ## State1.1.0: All necessary multiprocessing modules are imported
            from multiprocessing import get_context
            ## State1.1.1: For each of the tiles, the input parameter to the createPatch function is added to the list
            inputs = []
            for r in range(self.dim):
                for a in range(-self.size, self.size+1):
                    for s in range(r+1, self.dim):
                        for b in range(-self.size, self.size+1):
                            inputC = (r, a, s, b)
                            inputs.append(inputC)
            ## State1.1.2: A multiprocessing pool along with a context switch enables multiprocessing in python3 (accessed through dispPool)
            with get_context("spawn").Pool(os.cpu_count()-1) as dispPool:
                ## State1.1.2.0: A multiprocessing pool maps the createPatch function onto all the input parameters in inputs
                patches = [item for item in dispPool.imap(self.createPatch, inputs)]
                dispPool.close()
                dispPool.join()
            ## State1.1.3: All the patches are added to the figure for display
            self.patches = []
            for patch in patches:
                self.ax.add_patch(patch)
                self.patches.append(patch)
        else:
            ## State1.2.0: All necessary multithreading modules
            self.patches = []
            for r in range(self.dim):
                for a in range(-self.size, self.size+1):
                    for s in range(r+1, self.dim):
                        for b in range(-self.size, self.size+1):
                            ## Staet1.2.0.0: For all input parameters, the patch is constructed and added to the list of patches and figure linearly
                            inputC = (r, a, s, b)
                            patch = self.createPatch(inputC)
                            self.ax.add_patch(patch)
                            self.patches.append(patch)
        ## State2: If the pt is the original tiling, the axis is returned manually
        if animating and self.ptIndex==0:
            return self.ax
        ## State3: If the PA is not animated, the first figure is saved manually to the IO filestructure
        if not animating:
            self.saveFig()
        ## State4: If we get this far, the figure is automatically saved by an external function
        
    ## Display boundaries
    def displayBoundaries(self):
        self.setFigExtras()
        #for stablePatch in self.stablePatches:
            #self.ax.add_patch(stablePatch)
        self.patches = []
        ######## YOu can make it so that it does not need a sampColor, this reduces an extra valToBound and a few other extra calls
        for boundary in self.listOfBoundaryLists:
            samp = boundary[0]
            samp = self.multiGrid[samp[0]][samp[1]][samp[2]][samp[3]]
            sampColor = self.boundToCol.get(self.valToBound(samp.val))
            for index in boundary:
                t = self.multiGrid[index[0]][index[1]][index[2]][index[3]]
                if tuple(index) in self.invalidSet or len(t.neighbourhood) in self.borderSet:
                    continue
                else:
                    color = sampColor
                path = self.genPath(t.vertices)
                patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha) if self.tileOutline else mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
                self.ax.add_patch(patch)
                self.patches.append(patch)
        for index in self.specialTiles:
            if index in self.invalidSet:
                for i, invalidSet in enumerate(self.invalidSets):
                    if index in invalidSet:
                        color = self.invalidColors[i]
            else:
                color = self.borderColor
            t = self.multiGrid[index[0]][index[1]][index[2]][index[3]]
            path = self.genPath(t.vertices)
            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha) if self.tileOutline else mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
            self.ax.add_patch(patch)
            self.patches.append(patch)
    ## Display helper methods
    def setFigExtras(self):
        #ax.grid()
        lim = 10
        bound = lim*(self.size+self.dim+1)**1.2
        self.ax.set_xlim(-bound + self.zero[0], bound + self.zero[0])
        self.ax.set_ylim(-bound + self.zero[1], bound + self.zero[1])
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in self.shiftVect)
            self.ax.set_title(f"n={self.dim}, size={self.size}, shiftConstant={self.sC}, shiftVect~[{shiftVectStr}]")
        else:
            shifts = [str(round(i,1)) for i in self.shiftVect]
            self.ax.set_title(f"n={self.dim}, size={self.size}, shiftConstant={self.sC}, shiftVect~[{shifts[0]}, ..., {shifts[self.dim-1]}]")

    def saveFig(self):
        figPath = f'{self.rootPath}ByTimeStep/gen{self.ptIndex}'
        plt.savefig(figPath)

    #############################
    ## Non Implemented Methods ##
    #############################
    def selectiveDisplay(self, backgroundCol, selection, selectionCol):
        for tInd in self.allTiles:
            t = self.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
            path = self.genPath(t.vertices)
            tCol = selectionCol if tInd in selection else backgroundCol
            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = tCol, alpha=self.alpha)
            self.ax.add_patch(patch)
            self.patches.append(patch)
        return self.ax



###################
## Local Running ##
###################
def main():
    # dim up to 501 works for size = 1
    # dim = 5
    # sC = 0
    # size = 5
    # tileSize = 10
    # tileOutline = False
    # alpha = 1
    # shiftZeroes, shiftRandom, shiftByHalves = True, False, False
    # multi = Multigrid(dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, time.time())
    # multi.genTiling()
    # multi.genTileNeighbourhoods()
    # multi.displayTiling()
    pass

if __name__ == '__main__':
    main()