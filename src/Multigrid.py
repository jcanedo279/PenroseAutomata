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
                invalidSet=set(), invalidColor='purple', dispBorder=False, dispInvalid=True):
        
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

        ## Tile statistics
        self.numStable = 0
        self.numUnstable = 0
        self.numInvalid = 0
        self.stableTiles = []
        self.stablePatches = []
        self.unstableTiles = []
        self.invalidTiles = []

        self.invalidSet = invalidSet
        self.invalidColor = invalidColor
        self.dispBorder = dispBorder
        self.dispInvalid = dispInvalid

    #######################
    ## Generator Methods ##
    #######################
    def genNormVect(self):
        return [(-1)**((2/self.dim)*i) for i in range(self.dim)]
    
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

    ########################
    ## Conversion Methods ##
    ########################
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

    def valToBound(self, val):
        ## Search sorted is the perfect function for this application and does the job in O(log(n)) tc
        ## Search sorted returns the index of the first number in self.bounds that is greater then or equal to val
        return self.bounds[np.searchsorted(self.bounds, val)]
       
    ####################
    ## Setter Methods ##
    ####################
    def setValStats(self, numTiles, valAvg, valStdDev):
        self.numTiles = numTiles
        self.valAvg = valAvg
        self.valStdDev = valStdDev

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

    ########################
    ## Functional Methods ##
    ########################
    def saveFig(self):
        if not os.path.isdir(self.rootPath):
            os.makedirs(self.rootPath)
        plt.savefig(self.pathPng)

    def addToStablePatch(self, stableTiles):
        for stableTileInd in stableTiles:
            stableTile = self.multiGrid[stableTileInd[0]][stableTileInd[1]][stableTileInd[2]][stableTileInd[3]]
            color = self.boundToCol.get(self.valToBound(stableTile.val))
            if stableTile.val == -1:
                if self.dispInvalid:
                    color = self.invalidColor
                else:
                    return
            path = self.genPath(stableTile.vertices)
            if self.tileOutline:
                patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha)
            else:
                patch = mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
            self.stablePatches.append(patch)

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
        ## Even dimmensions
        if self.dim%2==0:
            lB = math.floor((self.dim/2)-1)
            sV = []
            for i in range(lB):
                sV.append((i+1)/self.dim)
                sV.append(-(i+1)/self.dim)
            sV.append((1/3)*self.sC)
            sV.append((2/3)*self.sC)
            return sV
        ## Odd dimmensions
        else:
            lB = math.floor((self.dim-1)/2)
            uB = math.ceil((self.dim-1)/2)
            sV = []
            popMultiplier = 1
            samplePopulation = list(range(1, popMultiplier*self.dim))
            samp = rd.sample(samplePopulation, lB)
            samp.sort()
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

    def genTilingVerts(self):
        self.multiGrid = [[[[None for b in range(-self.size, self.size+1)] for s in range(self.dim)] for a in range(-self.size, self.size+1)] for r in range(self.dim)]
        self.allTiles = []
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):

                        tileType = self.ttm[r][s]
                        self.allTiles.append(tuple([r, a, s, b]))
                        
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

                        ## Set the origin of the tiling, this ensures a properly centered animation
                        if(r==0 and s==1 and a==0 and b==0):
                            self.zero = [vertices[0][0], vertices[0][1]]

                        
                        if len(p_rs_ab.neighbourhood) in {0, 1, 2, 3, 4, 5, 6}:
                            p_rs_ab.setVal(-1)
                            p_rs_ab.setColor(self.invalidColor)
                            if self.dispBorder:
                                self.invalidTiles.append(p_rs_ab_ind)
                                self.numInvalid += 1
                        if p_rs_ab_ind in self.invalidSet:
                            p_rs_ab.setVal(-1)
                            p_rs_ab.setColor(self.invalidColor)
                            if self.dispInvalid:
                                self.invalidTiles.append(p_rs_ab_ind)
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
                        ## If the grid is not valued, value defaults to the tile type
                        elif not self.isValued:
                            p_rs_ab.setColor(self.boundToCol.get(tileType+1))
                            p_rs_ab.setVal(tileType)
                            self.values.append(tileType)
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
                            sampleDef = 1000
                            val=rd.randrange(tileType, (tileType+1)*sampleDef)/sampleDef
                            ## In order to avoid 0 and 1 being in same bound, add 1.
                            p_rs_ab.setColor(self.boundToCol.get(tileType+1))
                            p_rs_ab.setVal(val)
                            self.values.append(val)
                        ## Why do I need to do this?
                        self.multiGrid[r][a][s][b] = p_rs_ab
        self.numTiles = len(self.values)
        self.gridValAvg = sum(self.values)/self.numTiles
        valStdDev = 0
        for value in self.values:
            valStdDev += (value-self.gridValAvg)**2
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
                            invalidSet=self.invalidSet, invalidColor=self.invalidColor, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid)
        nextGrid.multiGrid, nextGrid.zero = self.multiGrid, self.zero
        nextGrid.values, nextGrid.numTiles, nextGrid.gridValAvg, nextGrid.valStdDev = self.values, self.numTiles, self.gridValAvg, self.valStdDev
        nextGrid.invalidTiles, nextGrid.numInvalid = self.invalidTiles, self.numInvalid
        
        self.values = []
        self.colValDict = {}
        self.stableTiles = []
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
        ## Iterate over the unstable tiles
        else:
            prevUnstableTiles = self.unstableTiles
            # if self.dispInvalid:
            #     for tile in self.invalidTiles:
            #         prevUnstableTiles.append(tile)
            self.unstableTiles = []
            for tInd in prevUnstableTiles:
                oldTile = self.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                newTile = nextGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                newTile.neighbour = oldTile.neighbourhood
                self.genNextTile(oldTile, newTile)



                
        self.unstableTiles = list(set(self.unstableTiles))
        self.numTiles = len(self.values)
        self.valAvg = sum(self.values)/self.numTiles
        self.valStdDev = math.sqrt(sum([(val-self.valAvg)**2 for val in self.values])/self.numTiles)
        if not self.gol:
            nextGrid.unstableTiles = self.unstableTiles
            nextGrid.invalidTiles = self.invalidTiles
            ##########################################
            ## This code creates a patch that does not need to be checked next round
            ## self.addToStablePatch(self.stableTiles)
            ##########################################
        nextGrid.stablePatches = self.stablePatches

        if boundaried:
            nextGrid.genBoundaryList()
            if self.ptIndex >= self.printGen:
                nextGrid.displayBoundaries()
        else:
            if self.ptIndex >= self.printGen:
                nextGrid.displayTiling()
        self.percentStable = self.numStable/self.numTilesInGrid
        self.percentUnstable = self.numUnstable/self.numTilesInGrid
        self.totalPercent = (self.numStable + self.numUnstable)/self.numTilesInGrid
        self.normPercentStable = self.numStable/(self.numStable+self.numUnstable)
        self.normPercentUnstable = self.numUnstable/(self.numStable+self.numUnstable)

        if animating:
            return nextGrid, self.ax
        return nextGrid

    def genNextTile(self, oldTile, newTile):
        if len(oldTile.neighbourhood) in {0, 1, 2, 3, 4, 5, 6}:
            newTile.setVal(-1)
            newTile.setColor(self.invalidColor)
        elif oldTile.val == -1:
            newTile.setVal(-1)
            newTile.setColor(self.invalidColor)
        elif self.gol:
            self.updateTileValsGOL(oldTile, newTile)
        else:
            self.updateTileVals(oldTile, newTile)
        if type(oldTile.color) is list:
            col = tuple(oldTile.color)
        col = tuple(oldTile.color) if type(oldTile.color) is list else oldTile.color
        ##### ADD UNSTABLE AND NON BOUDNARIED TILES HERE TO HAVE THEM SHOW UP IN COLOR COMP
        if col in self.colValDict:
            self.colValDict[col] += 1
        else:
            self.colValDict[col] = 1
        self.values.append(oldTile.val)

    ###########################################
    ## Generate Neighbourhoods For All Tiles ##
    ###########################################
    def genTileNeighbourhoods(self):
        ## For each tile, we give it a set of indices {[t.r, t.a, t.s, t.b],...} that can be used to iterate over
        ## to check the tile's local neighbourhood's values
        #
        ## We wish to make a dictionary w/ the keys containing all the vertices in Gngamma, where
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
                        for vertex in  t.vertices:
                            indexAsStr = f"{t.r} {t.a} {t.s} {t.b}"
                            if vertex in vertexToTileSet.keys():
                                tileSet = vertexToTileSet.get(vertex)
                                tileSet.add(indexAsStr)
                                vertexToTileSet.update({vertex : tileSet})
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
                if (not t.isStable) or (t.val==-1):
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
            ## Check if neighbours are invalid or unstable
            if (n.val==-1) or (nBound!=currBound):
                self.numUnstable += 1
                oldTileisStable = False
                oldTile.setStability(False)
                self.unstableTiles.append(oldTileIndex)
                break
        if oldTileisStable:
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
        
    ##########################
    ## Grid Display Methods ##
    ##########################
    ## Display all tiles
    def displayTiling(self):
        self.setFigExtras()
        self.patches = []
        colorList = sns.cubehelix_palette(self.numTileTypes, dark=0, light=0.8)
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        t = self.multiGrid[r][a][s][b]
                        vertices = t.vertices
                        path = self.genPath(vertices)
                        color = colorList[self.ttm[r][s]]
                        if t.val == -1 or len(t.neighbourhood) in {0, 1, 2, 3, 4, 5, 6}:
                            if self.dispInvalid:
                                color = self.invalidColor
                            else:
                                break
                        if self.gol:
                            if t.val==0:
                                color = 'white'
                            else:
                                color = 'black'
                        # The end
                        elif tuple([r, a, s, b]) in self.invalidSet:
                            if self.dispInvalid:
                                color = self.invalidColor
                            else:
                                break
                        # Adaptive Background
                        # elif abs(a)==self.size or abs(b)==self.size:
                        #     if self.ptIndex > 0:
                        #         bound = self.valToBound(self.gridValAvg)
                        #         ### THIS BECOMES EXPENSIVE
                        #         # boundInd = self.bounds.index(bound)
                        #         boundInd = np.searchsorted(self.bounds, bound)
                        #         if bound <= 1:
                        #             val = rd.randrange(0, bound+1)
                        #         else:
                        #             val = rd.randrange(self.bounds[boundInd-1], bound)
                        #         color = self.boundToCol.get(bound)
                        #         t.setColor(color)
                        #         t.setVal(val)
                        # Origin
                        # Playable Space
                        elif self.isValued:
                            color = t.color
                        if self.tileOutline:
                            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha)
                        else:
                            patch = mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
                        self.ax.add_patch(patch)
                        self.patches.append(patch)
        if self.ptIndex==0:
            return self.ax
    ## Display only boundaries
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
                if tuple(index) in self.invalidSet or len(t.neighbourhood) in {0, 1, 2, 3, 4, 5, 6}:
                    continue
                else:
                    color = sampColor
                path = self.genPath(t.vertices)
                patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha) if self.tileOutline else mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
                self.ax.add_patch(patch)
                self.patches.append(patch)
        for index in self.invalidTiles:
            t = self.multiGrid[index[0]][index[1]][index[2]][index[3]]
            path = self.genPath(t.vertices)
            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = self.invalidColor, alpha=self.alpha) if self.tileOutline else mpatches.PathPatch(path, edgecolor = self.invalidColor, facecolor = self.invalidColor, alpha=self.alpha)
            self.ax.add_patch(patch)
            self.patches.append(patch)



    
    def selectiveDisplay(self, backgroundCol, selection, selectionCol):
        for tInd in self.allTiles:
            t = self.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
            path = self.genPath(t.vertices)
            tCol = selectionCol if tInd in selection else backgroundCol
            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = tCol, alpha=self.alpha)
            self.ax.add_patch(patch)
            self.patches.append(patch)
        return self.ax



    def saveFigure(self):
        figPath = f'{self.rootPath}gen{self.ptIndex}.png'
        plt.savefig(figPath)







###################
## Local Running ##
###################
def main():
    # dim up to 501 works for size = 1
    dim = 5
    sC = 0
    size = 5
    tileSize = 10
    tileOutline = False
    alpha = 1
    shiftZeroes, shiftRandom, shiftByHalves = True, False, False
    multi = Multigrid(dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, time.time())
    multi.genTiling()
    multi.genTileNeighbourhoods()
    multi.displayTiling()

if __name__ == '__main__':
    main()