import time
import os
import glob
import shutil

import cProfile
import pstats

import math
import datetime

import itertools

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


## MultigridList Class
class MultigridList:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, numColors, minGen, maxGen, fitGen,
                 isValued=True, initialValue=(True, False, False, False), valRatio=None, numStates=None,
                 gol=False, overide=False, isBoundaried=False, boundaryReMap=False, boundaryApprox=True, shiftVect=None):
        
        ## Clear and init file space
        self.initFileSpace(cleaning=True, selectiveSpace=False, trashSpace=True)

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

        self.numTilesInGrid = (math.factorial(self.dim)/(2*(math.factorial(self.dim-2))))*((2*self.size+1)**2)

        ## States and colors
        self.numStates = numStates
        self.numColors = numColors

        ## Classic colors
        #self.colors = sns.color_palette("bright", self.numColors)
        self.colors = sns.color_palette("husl", self.numColors)
        #self.colors = sns.color_palette("cubehelix", self.numColors)

        ## Divergent colors
        #self.colors = sns.color_palette("BrBG", self.numColors)
        #self.colors = sns.color_palette("coolwarm", self.numColors)

        ## Gradient colors
        #self.colors = sns.cubehelix_palette(self.numColors, dark=0.1, light=0.9)

        ## State value settings
        self.isValued = isValued
        self.initialValue = initialValue
        self.valRatio=valRatio

        ## Boundary variables
        self.isBoundaried = isBoundaried
        self.boundaryReMap = boundaryReMap
        self.boundaryApprox = boundaryApprox
        self.percentStableRepeatCount = 0
        self.prevPercentStable = 1

        self.percentStables, self.percentUnstables, self.percentTotals = [], [], []
        self.normPercentStables, self.normPercentUnstables = [], []
        self.numBoundaries, self.numUnstable = [], []
        self.valAvgs, self.valStdDevs = [], []
        self.colValDicts = []

        ## Generation counters
        self.minGen = minGen
        self.maxGen = maxGen
        self.fitGen = fitGen

        ## Misc
        self.overide = overide

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
        self.currentGrid = Multigrid(self.dim, self.size, sC=self.sC, tileSize=self.tileSize, shiftProperties=shiftProperties,
                            tileOutline=self.tileOutline, alpha=self.alpha, rootPath=self.localTrashPath,
                            isValued=self.isValued, initialValue=self.initialValue,
                            bounds=self.bounds, boundToCol=self.boundToCol, boundToPC=self.boundToPC, boundaryApprox=self.boundaryApprox,
                            ptIndex=self.ptIndex, valRatio=self.valRatio, numStates=self.numStates, colors=self.colors,
                            gol=self.gol, isBoundaried=isBoundaried, shiftVect=self.shiftVect, numTilesInGrid=self.numTilesInGrid)

        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')

            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False)
            self.updateDir(self.currentGrid.shiftVect)

            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + self.currentGrid.zero[0], bound + self.currentGrid.zero[0])
            self.ax.set_ylim(-bound + self.currentGrid.zero[1], bound + self.currentGrid.zero[1])
        else:
            self.multigridList.append(self.currentGrid)
            self.tilingIter(self.currentGrid)
        self.saveAndExit()

    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridList[self.ptIndex-2].fig.clf()
        #    self.multigridList[self.ptIndex-2].ax.cla()

    def initFileSpace(self, cleaning, selectiveSpace=False, trashSpace=False):
        if cleaning:
            if selectiveSpace:
                files = glob.glob('MultigridListData/*')
                for f in files:
                    shutil.rmtree(f)
            if trashSpace:
                tFiles = glob.glob('TrashLists/*')
                for tF in tFiles:
                    shutil.rmtree(tF)

    ########################################
    ## Update Methods For Animation Cycle ##
    ########################################
    def updateDir(self, shiftVect):
        #if self.dim < 7:
            #shiftVectStr = ', '.join(str(round(i,1)) for i in shiftVect)
            #self.ax.set_title(f"n={self.dim}, size={self.size}, shiftConstant={self.sC}, shiftVect~[{shiftVectStr}]")
        #else:
            #shifts = [str(round(i,1)) for i in shiftVect]
            #self.ax.set_title(f"n={self.dim}, size={self.size}, shiftConstant={self.sC}, shiftVect~[{shifts[0]}, ..., {shifts[self.dim-1]}]")
        os.makedirs(self.localTrashPath)
        self.anim.save(self.gifTrashPath, writer=PillowWriter(fps=4))

    def updateAnimation(self, i):
        if self.ptIndex==0:
            self.currentGrid.genTiling()
            print('Original tile generated')
            self.currentGrid.genTileNeighbourhoods()
            print('Tile neighbourhood generated')
            axis = self.currentGrid.displayTiling()
            self.origAx = axis
        else:
            origGrid = self.currentGrid
            origGrid.ax.cla()
            nextGrid, axis = origGrid.genNextValuedGridState(animating=True, boundaried=self.isBoundaried)

            self.percentStables.append(origGrid.percentStable)
            self.percentUnstables.append(origGrid.percentUnstable)
            self.percentTotals.append(origGrid.totalPercent)
            self.normPercentStables.append(origGrid.normPercentStable)
            self.normPercentUnstables.append(origGrid.normPercentUnstable)
            self.valAvgs.append(origGrid.valAvg)
            self.valStdDevs.append(origGrid.valStdDev)
            self.colValDicts.append(origGrid.colorValDict)

            if self.isBoundaried:
                self.numBoundaries.append(nextGrid.numBoundaries)
                self.numUnstable.append(len(origGrid.unstableTiles))
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
            if self.ptIndex < self.minGen:
                pass
            elif self.isBoundaried and (not self.gol) and (not self.overide) and (self.ptIndex > 1) and ((nextGrid.numBoundaries/self.origNumBoundaries < 0.5) or (self.percentStableRepeatCount > 1)):
                self.continueAnimation = False
            self.currentGrid = nextGrid
            ## I think this helps with the memory leaks
            del origGrid
        ## Format animation title
        if self.dim < 7:
            shiftVectStr = ', '.join(str(round(i,1)) for i in self.currentGrid.shiftVect)
            title = f'n={self.dim}, size={self.size}, sC={self.sC}, sV=[{shiftVectStr}], gen={self.ptIndex}'
            self.ax.set_title(title)
        else:
            shifts = [str(round(i,1)) for i in self.currentGrid.shiftVect]
            title = f'n={self.dim}, size={self.size}, sC={self.sC}, sV=[{shifts[0]}, ..., {shifts[self.dim-1]}], gen={self.ptIndex}'
            self.ax.set_title(title)
        print(f'Grid {self.ptIndex} complete')
        self.ptIndex += 1
        return axis

    ############################
    ## Parameter Safety Check ##
    ############################
    def parameterSafetyCheck(self):
        print('-'*40)
        ## This ensures that all multigrid parameters are valid
        if self.dim<3:
            print(f"   --Dimmension '{self.dim}' < 3, dimmensions smaller than three cannot be projected")
            print('     Dimmension defaulted to 5')
            self.dim=3
        if self.size<0:
            print(f"   --Size '{self.size}' < 0, sizes the minimum size of a tiling is 0")
            print('     Size defaulted to 0')
            self.size=0
        if self.tileSize<0:
            print(f"  --Tiling size '{self.tileSize}' < 0, the size of each tile must be positive, as do all lengths")
            print('     Tiling Size deafulted to 10')
            self.tileSize=10
        if self.numColors>self.numStates:
            print(f"   --The number of colors '{self.numColors}' > the numebr of states '{self.numStates}', this is impossible")
            print('     The number of colors and the number of states both defaulted to 10')
            self.numStates, self.numColors = 10, 10
        if self.numColors<1:
            print(f"   --The number of colors '{self.numColors}' < 1, the number of colors must be greater than 0")
            print('     The number of colors defaulted to 10')
            if self.numStates>=10:
                self.numColors=10
            else:
                self.numColors=self.numStates
        if self.numStates<1:
            self.numStates=self.numColors
            print(f"   --The number of states '{self.numStates}' < 1, the number of states must be greater than 0")
            print(f"     The number of states defaulted to the number of states: '{self.numStates}'")
        if self.maxGen<0:
            self.maxGen=0
            print(f"   --The maximum generation '{self.maxGen}' < 0, the maximum generation must be greater than or equal to 0")
            print('     The maximum generation defaulted to 0')
        if self.gol==True and self.isBoundaried==True:
            print('   --The gamemode is set to GOL, yet isBoundaries=True')
            print('     isBoudnaried defaulted to False')
            self.isBoundaried=False
        if self.initialValue[0] and self.dim<11:
            print('         --he initial value condition of the space is a function of the tile type, itself a function of dim,')
            print(f"          which in this case is '{self.dim}' < 11.")
            print('          Please keep in mind that tileType ~dim/2. In order to increase complexity, try increasing dim to 11 or more')
        if self.initialValue[1] and self.dim<7:
            print(f"        --The initial value condition of the space is a function of dim, which in this case is '{self.dim}' < 7")
            print('          In order to increase complexity, try increasing dim to 7 or more (preferably 15+)')
        if self.initialValue[2] and self.size<5:
            print(f"        --The initial value condition of the space is a function of size, which in this case is '{self.size}' < 5")
            print('          Please keep in mind that part of the space may not be playable (ie the background).  In order to imcrease complexity,')
            print('          try increasing size to 5 or more (note that size is more than twice as costly as dimmension)')
        if self.initialValue[1]:
            print('-valuedByDim detected, numColors defaulted to dim')
            self.numColors = self.dim
        elif self.initialValue[2]:
            print('-valuedBySize detected, numColors defaulted to size+1')
            self.numColors = self.size+1
        elif self.initialValue[3]:
            print('-valuedByTileType detected, numColors and numStates defaulted to the number of types of tiles in the grid')
            if self.dim%2 == 0:
                self.numStates = int((self.dim/2)-1)
            else:
                self.numStates = int((self.dim-1)/2)
            self.numColors = self.numStates
            print(f' In this case, numColors and numStates both equal {self.numColors}')
        print('-'*40)
        print('All Paramters Validated')
        print('Parameter Safety Check Passed')
        print('-'*40)
        print(' '*40)
        print('-'*40)

    #######################
    ## Generator Methods ##
    #######################
    def genFrames(self):
        while (self.ptIndex < self.maxGen)  and self.continueAnimation:
            self.ax.cla()
            yield self.ptIndex
        yield StopIteration

    def genDirectoryPaths(self):
        ## rootPath and path data
        self.rootPath = 'MultigridListData/'
        self.multigridListInd = str(rd.randrange(0, 2000000000))
        self.gridPath = f'listInd{self.multigridListInd}/'
        ## localPaths
        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('MultigridListData', 'TrashLists')
        ## gifPaths
        self.gifPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}Animation.gif'
        self.gifTrashPath = self.gifPath.replace('MultigridListData', 'TrashLists')
        ## stabilityPaths
        self.stabilityPath, self.stabilityTrashPath = self.genPngPaths('stabilityStats')
        ## boundaryFigPaths
        self.boundaryPath, self.boundaryTrashPath = self.genPngPaths('boundaryStats')
        ## valPaths
        self.valPath, self.valTrashPath = self.genPngPaths('valStats')
        ## colorCompPaths
        self.colorCompPath, self.colorCompTrashPath = self.genPngPaths('colorCompStats')
        ## normColCompPaths
        self.normColCompPath, self.normColCompTrashPath = self.genPngPaths('normColColorCompStats')

    def genPngPaths(self, pngName):
        pngPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}{pngName}.png'
        return pngPath, pngPath.replace('MultigridListData', 'TrashLists')

    def genBounds(self):
        sampleDef = 1000
        upper = 1
        if self.boundaryReMap:
            upper = 2
        if self.numStates==1:
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
                if rd.randrange(0, upper) == 0:
                    self.boundToKeyCol[bound] = self.colors[i]
                else:
                    self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]

    ########################
    ## Save Final Results ##
    ########################
    def saveStabilityFig(self):
        stabilityFig = go.Figure()
        x = [str(i) for i in range(len(self.percentTotals))]
        stabilityFig.add_trace(go.Scatter(x=x, y=self.percentTotals, name='total percentage', line=dict(color='firebrick', width=4)))
        stabilityFig.add_trace(go.Scatter(x=x, y=self.percentStables, name='stable percentage', line=dict(color='firebrick', width=4, dash='dash')))
        stabilityFig.add_trace(go.Scatter(x=x, y=self.percentUnstables, name='unstable percentage', line=dict(color='firebrick', width=4, dash='dot')))
        stabilityFig.add_trace(go.Scatter(x=x, y=[1 for i in range(len(self.percentTotals))], name='normalized total percentage', line=dict(color='royalblue', width=4)))
        stabilityFig.add_trace(go.Scatter(x=x, y=self.normPercentStables, name='normalized stable percentage', line=dict(color='royalblue', width=4, dash='dash')))
        stabilityFig.add_trace(go.Scatter(x=x, y=self.normPercentUnstables, name='normalized unstable percentage', line=dict(color='royalblue', width=4, dash='dot')))
        stabilityFig.update_layout(title='Stability Statistics vs. Tile Generation',
                                  xaxis_title='tile Index',
                                  yaxis_title='percent composition of all tiles')
        stabilityFig.write_image(self.stabilityTrashPath)
    def saveBoundaryFig(self):
        ## First check if the figure is not boundaried
        boundaryFig = go.Figure()
        x = [str(i) for i in range(len(self.numBoundaries))]
        boundaryFig.add_trace(go.Scatter(x=x, y=self.numBoundaries, name='number of boundaries', line=dict(color='firebrick', width=4)))
        boundaryFig.add_trace(go.Scatter(x=x, y=self.numUnstable, name='number of unstable tiles', line=dict(color='royalblue', width=4)))
        boundaryFig.update_layout(title='Boundary Stats vs. Tile Generation',
                                  xaxis_title='tile Index',
                                  yaxis_title='number of tiles/boundaries')
        boundaryFig.write_image(self.boundaryTrashPath)
    def saveValFig(self):
        valFig = go.Figure()
        x = [str(i) for i in range(len(self.valAvgs))]
        valFig.add_trace(go.Scatter(x=x, y=self.valAvgs, name='grid value average', line=dict(color='firebrick', width=4)))
        valFig.add_trace(go.Scatter(x=x, y=self.valStdDevs, name='grid value std. dev.', line=dict(color='royalblue', width=4)))
        valFig.update_layout(title=f'Value Stats vs. Tile Generation',
                                  xaxis_title='tile Index',
                                  yaxis_title='statistic value')
        valFig.write_image(self.valTrashPath)
    def saveColorCompFig(self):
        colCompFig = go.Figure()
        x = [str(i) for i in range(len(self.colValDicts))]
        self.colComp = [[] for _ in range(len(self.colors))]
        for tilingInd in range(len(self.colValDicts)):
            for i, color in enumerate(self.colors):
                color = tuple(color) if type(color) is list else color
                if color in self.colValDicts[tilingInd]:
                    self.colComp[i].append(self.colValDicts[tilingInd].get(color))
                else:
                    self.colComp[i].append(0)
        for color, colHist, i in zip(self.colors, self.colComp, itertools.count()):
            color = tuple(color) if type(color) is list else color
            colCompFig.add_trace(go.Scatter(x=x, y=colHist, name=f'color{i}', line=dict(color=f'rgb{color}', width=4)))
        colCompFig.update_layout(title=f'Color Composition vs. Tile Generation',
                                  xaxis_title='tile Index',
                                  yaxis_title='number of tiles')
        colCompFig.write_image(self.colorCompTrashPath)
    def saveNormColCompFig(self):
        normColCompFig = go.Figure()
        x = [str(i) for i in range(len(self.colValDicts))]

        tilesPerGen = [percentTotal*self.numTilesInGrid for percentTotal in self.percentTotals]

        normColComp = self.colComp[:]
        for colorInd in range(len(self.colComp)):
            for tilingInd in range(len(self.colValDicts)):
                normColComp[colorInd][tilingInd] = normColComp[colorInd][tilingInd]/tilesPerGen[tilingInd]

        totalNumTiles = []
        for tilingInd in range(len(self.colValDicts)):
            numTiles = 0
            for colorInd in range(len(self.colComp)):
                numTiles += normColComp[colorInd][tilingInd]
            totalNumTiles.append(numTiles)

        for color, colHist, i in zip(self.colors, normColComp, itertools.count()):
            color = tuple(color) if type(color) is list else color
            normColCompFig.add_trace(go.Scatter(x=x, y=colHist, name=f'color{i}', line=dict(color=f'rgb{color}', width=4)))
        normColCompFig.add_trace(go.Scatter(x=x, y=totalNumTiles, name='total percentage', line=dict(color='black', width=4)))
        
        normColCompFig.update_layout(title=f'Normalized Color Composition vs. Tile Generation',
                                 xaxis_title='tile index',
                                 yaxis_title='percent composition of evaluated tiles')
        normColCompFig.write_image(self.normColCompTrashPath)

    #######################
    ## Save All And Exit ##
    #######################
    def saveAndExit(self):
        self.saveStabilityFig()
        self.saveValFig()
        if self.isBoundaried:
            self.saveBoundaryFig()
        if not self.gol:
            self.saveColorCompFig()
            self.saveNormColCompFig()
        ## Relocate fit tilings
        if self.ptIndex >= self.fitGen:
            files = glob.glob(self.localTrashPath)
            for f in files:
                shutil.move(f, self.localPath)
        ## Print execution completion and time
        print('-'*40)
        print(' '*40)
        print('-'*40)
        print('Algorithm Completed')
        print('Executed in {} seconds'.format(time.time()-self.startTime))
        print('-'*40)







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
## Main definition
def main():
    #p = cProfile.Profile()
    # Enable profiling
    #p.enable()

    dim = 5
    sC = 0 
    size = 19
    tileOutline = True
    alpha = 0.8
    shiftZeroes, shiftRandom, shiftByHalves = False, True, False
    shiftProperties = (shiftZeroes, shiftRandom, shiftByHalves)

    ## valuedRandomly, valuedByDim, valuedBySize, valuedByTT
    initialValue = (False, True, False, False)
    #valuedRandomly, valuedByDim, valuedBySize, valuedByTT = False, False, True, False

    numColors = 20
    numStates = 1000

    minGen = 20
    maxGen = 20
    fitGen = 40

    ## You can use this to overide any non properly tiled tilings
    shiftVect = None

    isBoundaried = True
    boundaryReMap = True
    ## Setting boundary approx trades time complexity for calculating the exact tiling
    ## Setting boundaryApprox as True improves time complexity and gives tiling approximation
    boundaryApprox = False

    ## Change gamemode to GOL
    gol = False

    ## Overide ensures maxGen generations
    overide = False

    ## This is the tileSize, dont mess around with this a lot
    tileSize = 10


    numIterations = 1
    for _ in range(numIterations):
        MultigridList(dim, sC, size, tileSize, shiftProperties, tileOutline, alpha, numColors, minGen, maxGen, fitGen,
                      isValued=True, initialValue=initialValue, valRatio=0.5, numStates=numStates,
                      gol=gol, overide=overide, isBoundaried=isBoundaried, boundaryReMap=boundaryReMap, boundaryApprox=boundaryApprox, shiftVect=shiftVect)

    #p.disable()
    #p.print_stats()
    # Dump the stats to a file
    #p.dump_stats("results.prof")

## Local main call
if __name__ == '__main__':
    main()



##### IF THE DIM IS EVEN, EACH ITERATION ITERATES OVER AN INVALID GRID INTERSECTION, THATS WHY WE GET SOME LINES NOT RHOMBS I THINK