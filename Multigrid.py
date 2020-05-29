import os

import random
import math
import itertools
import copy
import random as rd

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import seaborn as sns

import timeit



from MultigridCell import MultigridCell


## This class creates objects containing the parameters of a multigrid
## This randomly samples from all permutations of tiles T dual to the multigrid instance
class Multigrid:
    def __init__(self, dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, startTime=0, rootPath=None,
                rgb=False, isRadByDimrgb=False, isRadBySizergb=False, bounds=None, boundToCol=None, boundToPC=None,
                ptIndex=None, shiftVect=None, rgbRatio=None, numStates=None):


        ## Generate Multigrid object and instantiate its constant parameters
        self.startTime = startTime
        self.dim = dim
        self.sC = sC
        self.size = size
        self.tileSize = tileSize
        self.tileOutline = tileOutline
        self.alpha = alpha
        self.isrgbGrid, self.isRadByDimrgb, self.isRadBySizergb= rgb, isRadByDimrgb, isRadBySizergb
        self.rgbRatio = rgbRatio
        self.bounds, self.boundToCol, self.boundToPC = bounds, boundToCol, boundToPC
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftZeroes, shiftRandom, shiftByHalves
        self.numStates = numStates
        if ptIndex == 0:
            self.shiftVect = self.genShiftVector(zeroes=shiftZeroes, random=shiftRandom, byHalves=shiftByHalves)
        else:
            self.shiftVect = shiftVect
        ## adjacencyMatrix of tile types
        self.ttm = self.genttm()

        self.ax = plt.gca()

        self.rootPath = rootPath
        self.ptIndex = ptIndex
        self.gridPath = 'grid' + str(self.ptIndex) + '.png'
        self.pathPng = self.rootPath + self.gridPath


    def setrgbStats(self, numTiles, rgbAvg, rgbStdDev):
        self.numTiles = numTiles
        self.rgbAvg = rgbAvg
        self.rgbStdDev = rgbStdDev

    def setFigs(self, ax):
        self.ax = ax

    def genNextrgbGridState(self, animating=False):
        nextGrid = Multigrid(self.dim, self.sC, self.size, self.tileSize, self.shiftZeroes, self.shiftRandom, self.shiftByHalves, self.tileOutline, self.alpha,
                            rootPath=self.rootPath, bounds=self.bounds, boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                            rgb=True, isRadByDimrgb=self.isRadByDimrgb, isRadBySizergb=self.isRadBySizergb,
                            ptIndex=self.ptIndex+1, shiftVect=self.shiftVect, rgbRatio=self.rgbRatio, numStates=self.numStates)
        if animating:
            nextGrid.setFigs(self.ax)
        nextGrid.multiGrid = self.multiGrid
        nextGrid.zero = self.zero
        nextGrid.rgbs, nextGrid.numTiles, nextGrid.gridrgbAvg, nextGrid.stdDev = self.rgbs, self.numTiles, self.gridrgbAvg, self.stdDev

        rgbs = []
        for r in range(self.dim):
            ## We do not want to check the outermost layer as they are non playable
            for a in range(-self.size+1, self.size):
                for s in range(r+1, self.dim):
                    for b in range(-self.size+1, self.size):
                        t1 = self.multiGrid[r][a][s][b]
                        t2 = nextGrid.multiGrid[r][a][s][b]
                        t2.neighbourhood = t1.neighbourhood
                        if len(t2.neighbourhood) == 0:
                            t2.setrgbVal(0)
                            t2.setColor('black')
                            continue
                        self.updateTileVals(t1, t2)
                        if abs(a) or abs(b) < self.size-1:
                            rgbs.append(t1.rgb)
        self.numTiles = len(rgbs)
        self.rgbAvg = sum(rgbs)/self.numTiles
        stdDev = 0
        for rgb in rgbs:
            stdDev += (rgb-self.rgbAvg)**2
        self.stdDev = math.sqrt(stdDev/self.numTiles)
        nextGrid.displayTiling()
        if animating:
            return nextGrid, self.ax
        return nextGrid

    def updateTileVals(self, t1, t2):
        t1neighbourrgbTotal = 0
        for neighbour in t2.neighbourhood:
            n = self.multiGrid[neighbour[0]][neighbour[1]][neighbour[2]][neighbour[3]]
            t1neighbourrgbTotal += n.rgb
        t1neighbourrgbAvg = t1neighbourrgbTotal/len(t1.neighbourhood)

        bound = self.getBound(t1neighbourrgbAvg)
        pc = self.boundToPC.get(bound)

        t2rgb = t1.rgb + (t1neighbourrgbAvg - t1.rgb)*pc
        #t2rgb = self.rgbRatio*t1.rgb + (1-self.rgbRatio)*t1neighbourrgbAvg
        t2Color = ''


        t2Color = self.boundToCol.get(self.getBound(t2rgb))
        t2.setrgbVal(t2rgb)
        t2.setColor(t2Color)

    def getBound(self, rgbVal):
        if rgbVal in self.bounds:
            return rgbVal
        else:
            boundCopy = self.bounds.copy()
            boundCopy.append(rgbVal)
            boundCopy.sort()
            return self.bounds[boundCopy.index(rgbVal)]

    def genZetaVect(self):
        return [(-1)**((2/self.dim)*i) for i in range(self.dim)]

    def genShiftVector(self, zeroes=False, random=False, byHalves=True):
        if zeroes:
            shiftVect = [0.00001 for i in range(self.dim-1)]
            shiftVect.append(self.sC)
            return shiftVect
        
        if self.dim==3:
            samplePopulation = list(range(1, 1000, 1))
            sample = rd.sample(samplePopulation, 2)
            samp1, samp2 = sample[0]/2, sample[1]/2
            final = self.sC - samp1 - samp2
            return [samp1, samp2, final]

        if self.dim%2==0:
            # If even
            lB = math.floor((self.dim/2)-1)
            sV = []
            for i in range(lB):
                sV.append((i+1)/self.dim)
                sV.append(-(i+1)/self.dim)
            sV.append((1/3)*self.sC)
            sV.append((2/3)*self.sC)
            return sV
        elif self.dim%2!=0:
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

    def genttm(self):
        t = int((self.dim-1)/2)
        self.t = t
        ## create tile type adjacency matrix
        ttm = [[0 for x in range(self.dim)] for y in range(self.dim)]
        for x in range(self.dim):
            for y in range(1, t+1):
               
                if x+y >= self.dim:
                    ttm[x][(x+y)-self.dim] = y-1
                else:
                    ttm[x][x+y] = y-1
                ttm[x][x-y] = y-1
        return ttm

    def tileParamToVertices(self, r, s, a, b):
        ## Return a generator object of tile vertices
        zeta = self.genZetaVect()
        if zeta[s-r].imag == 0:
            z = 1j*(zeta[r]*(b-self.shiftVect[s]) - zeta[s]*(a-self.shiftVect[r])) / 0.0001
        else:
            z = 1j*(zeta[r]*(b-self.shiftVect[s]) - zeta[s]*(a-self.shiftVect[r])) / zeta[s-r].imag
        k = [0--((z/t).real+p)//1 for t, p in zip(zeta, self.shiftVect)]
        for k[r], k[s] in [(a, b), (a+1, b), (a+1, b+1), (a, b+1)]:
            yield sum(x*t for t, x in zip(zeta, k))

    ## Called once at start to generate tiling
    def tiling(self):
        ## Find all tiles to be projected from the multigrid parameters
        colorList = sns.cubehelix_palette(self.t, dark=0, light=0.8)
        #colorList = sns.cubehelix_palette("RdBu_r", self.t, dark=0, light=0.8)

        ## Create multigrid instance of cell objects
        p = None
        self.multiGrid = [[[[p for b in range(-self.size, self.size+1)] for s in range(self.dim)] for a in range(-self.size, self.size+1)] for r in range(self.dim)]
        self.rgbs = []
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):

                        tileType = self.ttm[r][s]
                        color = colorList[tileType]

                        p_rs_ab = MultigridCell(self.dim, r, s, a, b, self.shiftVect[r], self.shiftVect[s], tileType)

                        
                        ## Create first radial rgb grid by dimmension
                        if self.ptIndex==0 and self.isRadByDimrgb:
                            if r==0:
                                rgbVal=rd.randrange(0, self.bounds[r])
                            else:
                                rgbVal=rd.randrange(self.bounds[r-1], self.bounds[r])
                            tileColor = self.boundToCol.get(self.bounds[r])
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setrgbVal(rgbVal)
                            self.rgbs.append(rgbVal)
                        ## Create first radial rgb grid by size
                        if self.ptIndex==0 and self.isRadBySizergb:
                            if abs(a)==0:
                                rgbVal=rd.randrange(self.bounds[abs(a)], self.bounds[abs(a)+1])
                            else:
                                rgbVal=rd.randrange(self.bounds[abs(a)-1], self.bounds[abs(a)])
                            tileColor = self.boundToCol.get(self.bounds[abs(a)])
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setrgbVal(rgbVal)
                            self.rgbs.append(rgbVal)
                        ## Create first random rgb grid
                        elif self.ptIndex==0 and self.isrgbGrid:
                            rgbVal = rd.randrange(1, 256)
                            tileColor = ''
                            if rgbVal in self.bounds:
                                tileColor = self.boundToCol.get(rgbVal)
                            else:
                                tileColor = self.rgbToBound(rgbVal)
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setrgbVal(rgbVal)
                            self.rgbs.append(rgbVal)
                        # Might not be necessary
                        elif self.isrgbGrid:
                            rgbVal = rd.randrange(0, 256)
                            tileColor = self.rgbToBound(rgbVal)
                            p_rs_ab.setColor(tileColor)
                            p_rs_ab.setrgbVal(rgbVal)
                            self.rgbs.append(rgbVal)
                        ## Color end 
                        elif abs(a)==self.size-1 and abs(b)==self.size-1:
                            color = "black"

                        rhombus, rhombCopy = itertools.tee(self.tileParamToVertices(r, s, a, b))
                        vertices = list(self.imagToReal(rhombCopy))
                        ## Set the origin of the tiling if it exists
                        if(r==0 and s==1 and a==0 and b==0):
                            self.zero = [vertices[0], vertices[1]]
                            


                        vList = []
                        it = iter(vertices)
                        for x in it:
                            vList.append((x, next(it)))
                        p_rs_ab.setVertices(vList)

                        self.multiGrid[r][a][s][b] = p_rs_ab

                        yield rhombus, color

        self.numTiles = len(self.rgbs)
        self.gridrgbAvg = sum(self.rgbs)/self.numTiles
        stdDev = 0
        for rgb in self.rgbs:
            stdDev += (rgb-self.gridrgbAvg)**2
        self.stdDev = math.sqrt(stdDev/self.numTiles)

    def imagToReal(self, vertices):
        ## Return 
        for cord in vertices:
            scaledCord = self.tileSize*cord
            yield from (scaledCord.real, scaledCord.imag)

    def rgbToBound(self, rgbVal):
        boundCopy = self.bounds.copy()
        boundCopy.append(rgbVal)
        boundCopy.sort()
        return self.boundToCol.get(self.bounds[boundCopy.index(rgbVal)])

    def makeTiling(self, printImage=False):
        for rhombus, color in self.tiling():
            ## Rid warning
            rhombus, color = rhombus, color


    def displayTiling(self, printImage=False):
        # Does this have to be its own method?
        # Think about using average from previous generation
        self.patches = []
        colorList = sns.cubehelix_palette(self.t, dark=0, light=0.8)
        for r in range(self.dim):
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        t = self.multiGrid[r][a][s][b]
                        vertices = t.vertices
                        path = self.genPath(vertices)
                        color = colorList[self.ttm[r][s]]
                        # The end
                        if (abs(a)==self.size-1 and abs(b)==self.size-1) or t.rgb==0:
                            # The end
                            color = 'black'
                        # Background
                        #elif abs(a)==self.size-1 or abs(b)==self.size-1:
                            #rgbVal = self.gridrgbAvg
                            #color = self.rgbToBound(rgbVal)
                            #t.setColor(color)
                            #t.setrgbVal(rgbVal)
                        # Origin

                        # Playable Space
                        elif self.isrgbGrid:
                            color = t.color
                        if self.tileOutline:
                            patch = mpatches.PathPatch(path, edgecolor = None, facecolor = color, alpha=self.alpha)
                        else:
                            patch = mpatches.PathPatch(path, edgecolor = color, facecolor = color, alpha=self.alpha)
                        self.ax.add_patch(patch)
                        self.patches.append(patch)

        #ax.grid()
        lim = 10
        bound = lim*(self.size+self.dim+1)**1.2
        self.ax.set_xlim(-bound + self.zero[0], bound + self.zero[0])
        self.ax.set_ylim(-bound + self.zero[1], bound + self.zero[1])
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in self.shiftVect)
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shiftVectStr + "]")
        else:
            shifts = [str(round(i,1)) for i in self.shiftVect]
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shifts[0] + ", ..., " + shifts[self.dim-1]  + "]")
            
        #if not os.path.isdir(self.rootPath):
        #    os.makedirs(self.rootPath)
        #plt.savefig(self.pathPng)



## For each tile, we give it a set of indices {[t.r, t.a, t.s, t.b],...} that can be used to iterate over
    ## to check the tile's local neighbourhood's values
    def genTileNeighbourhoods(self):
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
                            if vertex in vertexToTileSet.keys():
                                tileSet = vertexToTileSet.get(vertex)
                                indexAsStr = "{} {} {} {}".format(t.r, t.a, t.s, t.b)
                                tileSet.add(indexAsStr)
                                vertexToTileSet.update({vertex : tileSet})
                            else:
                                neighbourhoodSet = set()
                                indexAsStr = "{} {} {} {}".format(t.r, t.a, t.s, t.b)
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


    def genPath(self, vertices):
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

        
        


def main():
    # dim up to 401 works for size = 2
    start = timeit.timeit()
    dim = 80
    sC = 0
    size = 5
    tileSize = 10
    tileOutline = False
    alpha = 1
    shiftZeroes, shiftRandom, shiftByHalves = False, True, False
    multi = Multigrid(dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, start)
    multi.makeTiling(printImage=False)
    multi.genTileNeighbourhoods()
    multi.displayTiling(printImage=True)
    

if __name__ == '__main__':
    main()