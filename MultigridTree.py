import os
import shutil
import glob
import math
import time
import datetime
import seaborn as sns
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import plotly.figure_factory as ff


from Multigrid import Multigrid

class MultigridTree:
    def __init__(self, dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, c, maxGen,
                 isValued=True, valIsRadByDim=False, valIsRadBySize=False, valRatio=None, numStates=None,
                 gol=False, overide=False, isBoundaried=False, shiftVect=None):

        self.isBoundaried = isBoundaried
        self.stability = []
        self.numBoundaries = []
        self.prevPercentStable = 1
        self.percentStableRepeatCount = 0

        self.startTime = time.time()
        self.dim, self.sC, self.size = dim, sC, size
        self.tileSize = tileSize
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftProperties
        self.shiftVect=shiftVect
        self.tileOutline, self.alpha = tileOutline, alpha

        self.isValued = isValued
        self.valRatio=valRatio

        self.numColors = c
        self.maxGen = maxGen
        self.numStates = numStates

        self.overide = overide
        self.setOriginal = True

        self.gol = gol

        #self.colors = sns.cubehelix_palette(self.numColors, dark=0.1, light=0.9)
        self.colors = sns.color_palette("bright", self.numColors)
        #self.colors = sns.diverging_palette(255, 133, l=60, n=self.numColors, center="dark")
        self.genBounds()
        ## List of multigrids
        self.multigridTree = []

        self.rootPath = 'MultigridTreeData/'

        self.multigridTreeInd = rd.randrange(0, 2000000000)
        self.gridPath = 'treeInd' + str(self.multigridTreeInd) + '/'

        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('MultigridTreeData', 'TrashTrees')

        self.gifPath = self.localPath + 'treeInd' + str(self.multigridTreeInd) + 'Animation.gif'
        self.gifTrashPath = self.gifPath.replace('MultigridTreeData', 'TrashTrees')
        self.stabilityPath = self.localPath + 'treeInd' + str(self.multigridTreeInd) + 'stability.png'
        self.stabilityTrashPath = self.stabilityPath.replace('MultigridTreeData', 'TrashTrees')
        self.numBoundariesPath = self.localPath + 'treeInd' + str(self.multigridTreeInd) + 'numBoundaries.png'
        self.numBoundariesTrashPath = self.numBoundariesPath.replace('MultigridTreeData', 'TrashTrees')

        self.ptIndex = 0

        origGrid = Multigrid(self.dim, self.size, sC=self.sC, tileSize=self.tileSize, shiftProperties=shiftProperties,
                            tileOutline=self.tileOutline, alpha=self.alpha, rootPath=self.localTrashPath,
                            isValued=self.isValued, valIsRadByDim=valIsRadByDim, valIsRadBySize=valIsRadBySize,
                            bounds=self.bounds, boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                            ptIndex=self.ptIndex, valRatio=self.valRatio, numStates=self.numStates, colors=self.colors,
                            gol=self.gol, isBoundaried=isBoundaried, shiftVect=self.shiftVect)
        self.animating = True
        self.continueAnimation = True
        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')
            origGrid.setFigs(self.ax)
        #self.ptIndex += 1

        origGrid.genTiling()
        print('Original tile generated')
        origGrid.genTileNeighbourhoods()
        print('Tile neighbourhood generated')
        self.origAx = origGrid.displayTiling()
        

        if self.animating:
            self.currentGrid = origGrid
        else:
            self.multigridTree.append(origGrid)
        
        self.numTilings = 1
        tilingIter = False

        if self.animating:
            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + origGrid.zero[0], bound + origGrid.zero[0])
            self.ax.set_ylim(-bound + origGrid.zero[1], bound + origGrid.zero[1])

            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False)
            self.finalOut(origGrid.shiftVect)  
        elif tilingIter:
            self.tilingIter(origGrid)

    def genFrames(self):
        while (self.ptIndex < self.maxGen-1)  and self.continueAnimation:
            self.ax.cla()
            yield self.ptIndex
        yield StopIteration

    def finalOut(self, shiftVect):
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in shiftVect)
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shiftVectStr + "]")
        else:
            shifts = [str(round(i,1)) for i in shiftVect]
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shifts[0] + ", ..., " + shifts[self.dim-1]  + "]")
        
        os.makedirs(self.localTrashPath)
        self.anim.save(self.gifTrashPath, writer=PillowWriter(fps=4))


    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridTree[self.ptIndex-2].fig.clf()
        #    self.multigridTree[self.ptIndex-2].ax.cla()

    def updateAnimation(self, i):
        if self.setOriginal:
            self.setOriginal=False
            return self.origAx
        if self.ptIndex==0:
            del self.origAx
        origGrid = self.currentGrid
        origGrid.ax.cla()
        nextGrid, axis = origGrid.genNextValuedGridState(animating=True, boundaried=self.isBoundaried)
        self.stability.append(origGrid.percentStable)
        if self.isBoundaried:
            self.numBoundaries.append(nextGrid.numBoundaries)
            if self.ptIndex==1:
                self.origNumBoundaries = nextGrid.numBoundaries
        if self.isBoundaried and (self.ptIndex > 1):
            if nextGrid.numBoundaries/self.origNumBoundaries == self.prevPercentStable:
                self.percentStableRepeatCount += 1
            else:
                self.percentStableRepeatCount = 0
            self.prevPercentStable = nextGrid.numBoundaries/self.origNumBoundaries
            #print(nextGrid.numBoundaries/self.origNumBoundaries, origGrid.numBoundaries, self.percentStableRepeatCount)
        if self.isBoundaried and (not self.gol) and (not self.overide) and (self.ptIndex > 1) and ((nextGrid.numBoundaries/self.origNumBoundaries < 0.5) or (self.percentStableRepeatCount > 1)):
            self.continueAnimation = False

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
        ## Lets see if this here helps w the memory leaks
        del origGrid
        self.numTilings += 1
        print('Grid {} complete'.format(self.ptIndex))
        return axis

    def genNextGrid(self, origGrid):
        nextGrid = origGrid.genNextValuedGridState()
        self.ptIndex += 1
        self.multigridTree.append(nextGrid)
        self.numTilings += 1
        return nextGrid
                        
    def tilingIter(self, rootPop):
        currentPop = rootPop
        while(self.ptIndex < self.maxGen):
            currentPop = self.genNextGrid(currentPop)

    def genBounds(self):
        sampleDef = 1000
        if self.numStates==0:
            print('INVALID NUMBER OF STATES')
        elif self.numStates==1:
            self.bounds = [self.numStates]
            self.boundToCol, self.boundToKeyCol, self.boundToPC = {}, {}, {}
            self.boundToCol[0] = self.colors[0]
            self.boundToKeyCol = self.boundToCol
            self.boundToPC[0] = rd.randrange(0, sampleDef+1)/sampleDef
        else:
            samplePop = range(1, self.numStates)
            sample = rd.sample(samplePop, self.numColors-1)
            sample.append(self.numStates)
            self.bounds = sorted(sample)
            self.boundToCol = {}
            self.boundToKeyCol = {}
            self.boundToPC = {}
            for i, bound in enumerate(self.bounds):
                self.boundToCol[bound] = self.colors[i]
                self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
                if rd.randrange(0, 2) == 0:
                    self.boundToKeyCol[bound] = self.colors[i]
                else:
                    self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]



def main():
    cleaning = True
    if cleaning:
        #files = glob.glob('MultigridTreeData/*')
        #for f in files:
            #shutil.rmtree(f)
        tFiles = glob.glob('TrashTrees/*')
        for tF in tFiles:
            shutil.rmtree(tF)

    numIterations = 1
    for _ in range(numIterations):
        dim = 20
        sC = 1/2
        size = 2
        tileOutline = True
        alpha = 1
        shiftZeroes, shiftRandom, shiftByHalves = True, False, False
        shiftProperties = (shiftZeroes, shiftRandom, shiftByHalves)
        ## For extreme tilings:
        #    If dim>>size -> isRadByDim=True
        #    If size>>dim -> isRadBySize=True
        isRadByDim, isRadBySize = True, False

        numColors = 1000
        numStates = 10000
        minGen = 5
        maxGen = 5

        shiftVect=None

        isBoundaried = False

        gol=False

        overide=False


        if isRadByDim:
            numColors = dim
        elif isRadBySize:
            numColors = size+1
        tileSize = 10
        tree = MultigridTree(dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, numColors, maxGen,
                             isValued=True, valIsRadByDim=isRadByDim, valIsRadBySize=isRadBySize, numStates=numStates,
                             gol=gol, overide=overide, isBoundaried=isBoundaried, shiftVect=shiftVect)
        
        ## Save the distributions of stable tiles
        group_labels_stdDev = ['percentage of stable tiles'] # name of the dataset
        hist_data_stdDev = [tree.stability]
        stdDevFig = ff.create_distplot(hist_data_stdDev, group_labels_stdDev, show_hist=False)
        stdDevFig.write_image(tree.stabilityTrashPath)
        ## Save the boundaries
        if tree.isBoundaried:
            group_labels_nmBnds = ['number of boundries']
            hist_data_nmBnds = [tree.numBoundaries]
            numBoundriesFig = ff.create_distplot(hist_data_nmBnds, group_labels_nmBnds, show_hist=False)
            numBoundriesFig.write_image(tree.numBoundariesTrashPath)
        ## Save fit tilings
        if tree.ptIndex > minGen:
            files = glob.glob(tree.localTrashPath)
            for f in files:
                shutil.move(f, tree.localPath)
        
        print('Algorithm Completed')
        print('Executed in {} seconds'.format(time.time()-tree.startTime))
        

if __name__ == '__main__':
    main()