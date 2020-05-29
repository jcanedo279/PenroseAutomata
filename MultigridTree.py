import math
import time
import datetime
import seaborn as sns
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


from Multigrid import Multigrid

class MultigridTree:
    def __init__(self, dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, c, maxGen,
    isrgb=True, isRadByDimrgb=False, isRadBySizergb=False, rgbRatio=None, numStates=None):
        self.startTime = time.time()
        self.dim, self.sC, self.size = dim, sC, size
        self.tileSize = tileSize
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftZeroes, shiftRandom, shiftByHalves
        self.tileOutline, self.alpha = tileOutline, alpha

        self.isrgb = isrgb
        self.rgbRatio=rgbRatio

        self.stdDev = float('inf')  

        self.numColors = c
        self.maxGen = maxGen
        self.numStates = numStates

        #self.colors = sns.cubehelix_palette(self.numColors, dark=0.1, light=0.9)
        self.colors = sns.color_palette("bright", self.numColors)
        #self.colors = sns.diverging_palette(255, 133, l=60, n=self.numColors, center="dark")
        self.genBounds()
        ## List of multigrids
        self.multigridTree = []

        self.rootPath = 'MultigridTreeFigs/'
        self.multigridTreeInd = rd.randrange(0, 2000000000)
        self.gridPath = 'treeInd' + str(self.multigridTreeInd) + '/'
        self.localPath = self.rootPath + self.gridPath
        self.gifPath = 'MultigridTreeGifs/' + self.gridPath

        self.ptIndex = 0

        origGrid = Multigrid(self.dim, self.sC, self.size, self.tileSize, self.shiftZeroes, self.shiftRandom,
                            self.shiftByHalves, self.tileOutline, self.alpha, rootPath=self.localPath,
                            rgb=self.isrgb, isRadByDimrgb=isRadByDimrgb, isRadBySizergb=isRadBySizergb,
                            bounds=self.bounds, boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                            ptIndex=self.ptIndex, rgbRatio=self.rgbRatio, numStates=self.numStates)
        self.animating = True
        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')
            origGrid.setFigs(self.ax)
        #self.ptIndex += 1

        origGrid.makeTiling(printImage=False)
        print('Original tile generated')
        origGrid.genTileNeighbourhoods()
        print('Tile neighbourhood generated')
        #origGrid.displayTiling()

        if not self.animating:
            self.multigridTree.append(origGrid)
        else:
            self.currentGrid = origGrid
        
        self.numTilings = 1

        if self.animating:
            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + origGrid.zero[0], bound + origGrid.zero[0])
            self.ax.set_ylim(-bound + origGrid.zero[1], bound + origGrid.zero[1])
            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.maxGen-1, init_func=self.initPlot(), interval=20)
            if self.dim < 7:
                shiftVectStr = ', '.join(str(round(i,1)) for i in origGrid.shiftVect)
                self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shiftVectStr + "]")
            else:
                shifts = [str(round(i,1)) for i in origGrid.shiftVect]
                self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shifts[0] + ", ..., " + shifts[self.dim-1]  + "]")
            
            
            gridPath = self.gifPath[:-1] + 'Anmiation.gif'

            self.anim.save(gridPath, writer=PillowWriter(fps=5))
            
            print('Algorithm Completed')
            print('Executed in {} seconds'.format(time.time()-self.startTime))
        else:
            self.tilingIter(origGrid)
            plt.show()

    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridTree[self.ptIndex-2].fig.clf()
        #    self.multigridTree[self.ptIndex-2].ax.cla()

    def updateAnimation(self, i):

        origGrid = self.currentGrid
        origGrid.ax.cla()
        nextGrid, patches = origGrid.genNextrgbGridState(animating=True)  
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in origGrid.shiftVect)
            title = 'n={}, size={}, sC={}, sV=[{}], gen={}'.format(self.dim, self.size, self.sC, shiftVectStr, self.ptIndex)
            self.ax.set_title(title)
        else:
            shifts = [str(round(i,1)) for i in origGrid.shiftVect]
            title = 'n={}, size={}, sC={}, sV=[{}, ..., {}], gen={}'.format(self.dim, self.size, self.sC, shifts[0], shifts[self.dim-1], self.ptIndex)
            self.ax.set_title(title)

        self.ptIndex += 1
        self.currentGrid = nextGrid
        self.numTilings += 1
        print('Grid {} complete'.format(self.ptIndex))
        return patches

    def genNextGrid(self, origGrid):
        nextGrid = origGrid.genNextrgbGridState()
        self.ptIndex += 1
        self.multigridTree.append(nextGrid)
        self.numTilings += 1
        return nextGrid
                        
    def tilingIter(self, rootPop):
        currentPop = rootPop
        while(self.ptIndex < self.maxGen):
            currentPop = self.genNextGrid(currentPop)

    def genBounds(self):
        samplePop = range(1, self.numStates)
        sample = rd.sample(samplePop, self.numColors-1)
        sample.append(self.numStates)
        self.bounds = sorted(sample)
        self.boundToCol = {}
        self.boundToKeyCol = {}
        self.boundToPC = {}
        sampleDef = 1000
        for i, bound in enumerate(self.bounds):
            self.boundToCol[bound] = self.colors[i]
            self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
            if rd.randrange(0, 2) == 1:
                self.boundToKeyCol[bound] = self.colors[i]
            else:
                self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]



def main():
    dim = 12
    sC = 0
    size = 5
    tileOutline = False
    alpha = 1
    shiftZeroes, shiftRandom, shiftByHalves = False, True, False
    isRadByDim, isRadBySize = True, False
    numColors = 10
    numStates = 255
    if isRadByDim:
        numColors = dim
    elif isRadBySize:
        numColors = size+1
    tileSize = 10
    maxGen = 150
    tree = MultigridTree(dim, sC, size, tileSize, shiftZeroes, shiftRandom, shiftByHalves, tileOutline, alpha, numColors, maxGen,
                         isrgb=True, isRadByDimrgb=isRadByDim, isRadBySizergb=isRadBySize, numStates=numStates)
    tree = tree

if __name__ == '__main__':
    main()