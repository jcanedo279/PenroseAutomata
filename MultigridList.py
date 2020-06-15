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

class MultigridList:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, c, minGen, maxGen,
                 isValued=True, valIsRadByDim=False, valIsRadBySize=False, valRatio=None, numStates=None,
                 gol=False, overide=False, isBoundaried=False, shiftVect=None):
        
        ## Animation on
        self.animating = True
        ## Early animation exit param
        self.continueAnimation = True

        ## Constant grid parameters
        self.startTime = time.time()
        self.dim, self.sC, self.size = dim, sC, size
        self.tileSize = tileSize
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftProperties
        self.shiftVect=shiftVect
        self.tileOutline, self.alpha = tileOutline, alpha

        ## States and colors
        self.numStates = numStates
        self.numColors = c
        self.colors = sns.color_palette("bright", self.numColors)
        #self.colors = sns.cubehelix_palette(self.numColors, dark=0.1, light=0.9)
        #self.colors = sns.diverging_palette(255, 133, l=60, n=self.numColors, center="dark")

        ## State value settings
        self.isValued = isValued
        self.valRatio=valRatio

        ## Boundary variables
        self.isBoundaried = isBoundaried
        self.numBoundaries = []
        self.percentStableRepeatCount = 0
        self.stability = []
        self.prevPercentStable = 1

        ## Generation counters
        self.minGen = minGen
        self.maxGen = maxGen

        ## Misc
        self.overide = overide
        self.setOriginal = True

        ## Game of life boolean
        self.gol = gol

        ## Generate directory names and paths
        self.genDirectoryPaths()

        ## Parameter Safety Check
        self.parameterSafetyCheck()

        ## Generate ruleset
        self.genBounds()

        ## Current tile count
        self.ptIndex = 0

        ## List that stores all multigrids, will by dynamically managed
        self.multigridList = []

        ## Create and animate original tiling
        origGrid = Multigrid(self.dim, self.size, sC=self.sC, tileSize=self.tileSize, shiftProperties=shiftProperties,
                            tileOutline=self.tileOutline, alpha=self.alpha, rootPath=self.localTrashPath,
                            isValued=self.isValued, valIsRadByDim=valIsRadByDim, valIsRadBySize=valIsRadBySize,
                            bounds=self.bounds, boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                            ptIndex=self.ptIndex, valRatio=self.valRatio, numStates=self.numStates, colors=self.colors,
                            gol=self.gol, isBoundaried=isBoundaried, shiftVect=self.shiftVect)
        self.ptIndex += 1

        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')
            origGrid.setFigs(self.ax)
        
        origGrid.genTiling()
        print('Original tile generated')
        origGrid.genTileNeighbourhoods()
        print('Tile neighbourhood generated')
        self.origAx = origGrid.displayTiling()
        
        if self.animating:
            self.currentGrid = origGrid
        else:
            self.multigridList.append(origGrid)

        if self.animating:
            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + origGrid.zero[0], bound + origGrid.zero[0])
            self.ax.set_ylim(-bound + origGrid.zero[1], bound + origGrid.zero[1])

            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False)
            self.updateDir(origGrid.shiftVect)  
        else:
            self.tilingIter(origGrid)

    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridList[self.ptIndex-2].fig.clf()
        #    self.multigridList[self.ptIndex-2].ax.cla()

    ########################################
    ## Update Methods For Animation Cycle ##
    ########################################
    def updateDir(self, shiftVect):
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in shiftVect)
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shiftVectStr + "]")
        else:
            shifts = [str(round(i,1)) for i in shiftVect]
            self.ax.set_title("n=" + str(self.dim) +  ", size=" + str(self.size) + ", shiftConstant=" + str(self.sC) + ",  shiftVect~[" + shifts[0] + ", ..., " + shifts[self.dim-1]  + "]")
        
        os.makedirs(self.localTrashPath)
        self.anim.save(self.gifTrashPath, writer=PillowWriter(fps=4))

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
        ## Stop the animation before maxGen
        if self.isBoundaried and (not self.gol) and (not self.overide) and (self.ptIndex > 1) and ((nextGrid.numBoundaries/self.origNumBoundaries < 0.5) or (self.percentStableRepeatCount > 1)):
            self.continueAnimation = False
        ## Format animation title
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
        ## I think this helps with the memory leaks
        del origGrid
        print('Grid {} complete'.format(self.ptIndex))
        return axis

    ############################
    ## Parameter Safety Check ##
    ############################
    def parameterSafetyCheck(self):
        ## This ensures that all multigrid parameters are valid
        if self.dim<3:
            self.dim=3
            print("Dimmension '{}' < 3, dimmensions smaller than three cannot be projected".format(self.dim))
            print('Dimmension defaulted to 5')
        if self.size<0:
            self.size=0
            print("Size '{}' < 0, sizes the minimum size of a tiling is 0".format(self.size))
            print('Size defaulted to 0')
        if self.tileSize<0:
            self.tileSize=10
            print("Tiling size '{}' < 0, the size of each tile must be positive, as do all lengths".format(self.tileSize))
            print('Tiling Size deafulted to 10')
        if self.numColors>self.numStates:
            self.numStates, self.numColors = 10, 10
            print("The number of colors '{}' > the numebr of states '{}', this is impossible".format(self.numColors, self.numStates))
            print('The number of colors and the number of states both defaulted to 10')
        if self.numColors<1:
            if self.numStates>=10:
                self.numColors=10
            else:
                self.numColors=self.numStates
            print("The number of colors '{}' < 1, the number of colors must be greater than 0".format(self.numColors))
            print('The number of colors defaulted to 10')
        if self.numStates<1:
            self.numStates=self.numColors
            print("The number of states '{}' < 1, the number of states must be greater than 0".format(self.numStates))
            print("The number of states defaulted to the number of states: '{}'".format(self.numStates))
        if self.maxGen<0:
            self.maxGen=0
            print("The maximum generation '{}' < 0, the maximum generation must be greater than or equal to 0")
            print('The maximum generation defaulted to 0')
        print('Paramters Validated')

    #######################
    ## Generator Methods ##
    #######################
    def genFrames(self):
        while (self.ptIndex < self.maxGen-1)  and self.continueAnimation:
            self.ax.cla()
            yield self.ptIndex
        yield StopIteration

    def genDirectoryPaths(self):
        self.rootPath = 'MultigridListData/'
        self.multigridListInd = rd.randrange(0, 2000000000)
        self.gridPath = 'listInd' + str(self.multigridListInd) + '/'
        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('MultigridListData', 'TrashLists')
        self.gifPath = self.localPath + 'listInd' + str(self.multigridListInd) + 'Animation.gif'
        self.gifTrashPath = self.gifPath.replace('MultigridListData', 'TrashLists')
        self.stabilityPath = self.localPath + 'listInd' + str(self.multigridListInd) + 'stability.png'
        self.stabilityTrashPath = self.stabilityPath.replace('MultigridListData', 'TrashLists')
        self.numBoundariesPath = self.localPath + 'listInd' + str(self.multigridListInd) + 'numBoundaries.png'
        self.numBoundariesTrashPath = self.numBoundariesPath.replace('MultigridListData', 'TrashLists')

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
            self.boundToCol, self.boundToKeyCol, self.boundToPC = {}, {}, {}
            for i, bound in enumerate(self.bounds):
                self.boundToCol[bound] = self.colors[i]
                self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
                if rd.randrange(0, 2) == 0:
                    self.boundToKeyCol[bound] = self.colors[i]
                else:
                    self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]

    ########################
    ## Save Final Results ##
    ########################
    ## Save stability figure
    def saveStabilityFig(self):
        group_labels_stdDev = ['percentage of stable tiles'] # name of the dataset
        hist_data_stdDev = [self.stability]
        stdDevFig = ff.create_distplot(hist_data_stdDev, group_labels_stdDev, show_hist=False)
        stdDevFig.write_image(self.stabilityTrashPath)
    ## Save boundary figure
    def saveBoundaryFig(self):
        ## First check if the figure is not boundaried
        if not self.isBoundaried:
            return
        group_labels_nmBnds = ['number of boundries']
        hist_data_nmBnds = [self.numBoundaries]
        numBoundriesFig = ff.create_distplot(hist_data_nmBnds, group_labels_nmBnds, show_hist=False)
        numBoundriesFig.write_image(self.numBoundariesTrashPath)
    ## Save figures and finalzie
    def finalizeAutomata(self):
        ## Save the distributions of stable tiles
        self.saveStabilityFig()
        ## Save the boundaries
        self.saveBoundaryFig()
        ## Relocate fit tilings
        if self.ptIndex > self.minGen:
            files = glob.glob(self.localTrashPath)
            for f in files:
                shutil.move(f, self.localPath)
        ## Print execution completion and time
        print('Algorithm Completed')
        print('Executed in {} seconds'.format(time.time()-self.startTime))


    #####################################################
    ## Old Methods For Generating Animation By Picture ##
    #####################################################
    def genNextGrid(self, origGrid):
        nextGrid = origGrid.genNextValuedGridState()
        self.ptIndex += 1
        self.multigridList.append(nextGrid)
        self.ptIndex += 1
        return nextGrid
                        
    def tilingIter(self, rootPop):
        currentPop = rootPop
        while(self.ptIndex < self.maxGen):
            currentPop = self.genNextGrid(currentPop)
    ######################################################
    ######################################################
    ######################################################









###############################
## Local Animation Execution ##
###############################
## Clean Multigrid directory
def cleanFileSpace(cleaning, selectiveSpace=False, trashSpace=False):
    if cleaning:
        if selectiveSpace:
            files = glob.glob('MultigridListData/*')
            for f in files:
                shutil.rmtree(f)
        if trashSpace:
            tFiles = glob.glob('TrashLists/*')
            for tF in tFiles:
                shutil.rmtree(tF)
## Main definition
def main():
    cleanFileSpace(cleaning=True, selectiveSpace=False, trashSpace=True)

    numIterations = 5
    for _ in range(numIterations):
        dim = 5
        sC = 1/2
        size = 5
        tileOutline = True
        alpha = 1
        shiftZeroes, shiftRandom, shiftByHalves = True, False, False
        shiftProperties = (shiftZeroes, shiftRandom, shiftByHalves)
        ## For extreme tilings:
        #    If dim>>size -> isRadByDim=True
        #    If size>>dim -> isRadBySize=True
        isRadByDim, isRadBySize = False, False

        numColors = 1000
        numStates = 10000
        minGen = 40
        maxGen = 40

        shiftVect=None

        isBoundaried = False

        gol=False

        overide=False

        if isRadByDim:
            numColors = dim
        elif isRadBySize:
            numColors = size+1
        tileSize = 10
        mList = MultigridList(dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, numColors, minGen, maxGen,
                             isValued=True, valIsRadByDim=isRadByDim, valIsRadBySize=isRadBySize, numStates=numStates,
                             gol=gol, overide=overide, isBoundaried=isBoundaried, shiftVect=shiftVect)
        ## Finalize tiling
        mList.finalizeAutomata()
## Local main call
if __name__ == '__main__':
    main()