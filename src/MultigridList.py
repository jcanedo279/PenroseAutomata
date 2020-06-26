import time
import os
import glob
import shutil

import cProfile
import pstats

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


## MultigridList Class
class MultigridList:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, size, shiftVect=None, sC=0, shiftProp=(False,True,True),
                 minGen=8, maxGen=10, fitGen=10,
                 numColors=10, manualCols=True, numStates=100,
                 isValued=True, initialValue=(True,False,False,False), valRatio=None,
                 isBoundaried=False, boundaryReMap=False, boundaryApprox=False,
                 gol=False, tileOutline=False, alpha=1, overide=False, printGen=0,
                 borderSet={1,2,3,4,5,6}, invalidSets=[], borderColor='black', invalidColors=[],
                 borderVal=-1, invalidVals=[], dispBorder=False, dispInvalid=True):
        
        ## Animation on
        self.animating = False
        ## Early animation exit param
        self.continueAnimation = True

        ## Constant grid parameters
        self.startTime = time.time()

        self.dim, self.size, self.sC = dim, size, sC
        self.shiftVect=shiftVect
        self.shiftProp = shiftProp
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = shiftProp

        ## Generation counters
        self.minGen, self.maxGen, self.fitGen, self.printGen = minGen, maxGen, fitGen, printGen

        ## States and colors
        self.numStates, self.numColors, self.manualCols = numStates, numColors, manualCols

        ## State value settings
        self.isValued, self.initialValue, self.valRatio = isValued, initialValue, valRatio

        ## Boundary variables
        self.isBoundaried, self.boundaryReMap, self.boundaryApprox = isBoundaried, boundaryReMap, boundaryApprox

        ## Game of life gamemode
        self.gol = gol
        self.tileOutline, self.alpha, = tileOutline, alpha
        self.overide = overide

        ## The number of tiles in this grid
        self.numTilesInGrid = (math.factorial(self.dim)/(2*(math.factorial(self.dim-2))))*((2*self.size+1)**2)

        ## Do not mess with this
        self.tileSize = 10

        self.borderSet, self.invalidSets, self.borderColor, self.invalidColors = borderSet, invalidSets, borderColor, invalidColors
        self.borderVal, self.invalidVals = borderVal, invalidVals
        self.dispBorder, self.dispInvalid = dispBorder, dispInvalid
        
        self.invalidSet = set()
        for invalidSet in self.invalidSets:
            self.invalidSet.update(invalidSet)

        ## Parameter Safety Check
        self.parameterSafetyCheck()

        ## Choose a color scheme
        self.genColors()

        ## Keep track of stability
        self.percentStableRepeatCount = 0
        self.prevPercentStable = 1

        ## Generate directory names and paths
        self.genDirectoryPaths()
        
        ## Generate lists used to keep track of statistics
        if self.animating:
            ## Generate ruleset
            self.genBounds()
            
            if self.isBoundaried:
                self.numBoundaries, self.numUnstable = [], []
                
                
            self.percentStables, self.percentUnstables, self.percentTotals = [], [], []
            self.normPercentStables, self.normPercentUnstables = [], []
            self.valAvgs, self.valStdDevs = [], []
            self.colValDicts = []

        ## Current tile count
        self.ptIndex = 0

        ## Create and animate original tiling
        self.currentGrid = Multigrid(self.dim, self.size, shiftVect=self.shiftVect, sC=self.sC, shiftProp=shiftProp,
                                     numTilesInGrid=self.numTilesInGrid,
                                     startTime=self.startTime, rootPath=self.localTrashPath, ptIndex=self.ptIndex, 
                                     numStates=self.numStates, colors=self.colors,
                                     isValued=self.isValued, initialValue=self.initialValue,  valRatio=self.valRatio,
                                     boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                                     isBoundaried=isBoundaried, bounds=self.bounds, boundaryApprox=self.boundaryApprox,
                                     gol=self.gol, tileOutline=self.tileOutline, alpha=self.alpha, printGen=self.printGen,
                                     borderSet=self.borderSet, invalidSets=self.invalidSets, borderColor=self.borderColor, invalidColors=self.invalidColors,
                                     borderVal=self.borderVal, invalidVals=self.invalidVals, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid)
        self.currentGrid.invalidSet = self.invalidSet

        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')

            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False)
            self.updateDir(self.currentGrid.shiftVect)
            self.saveTilingInfo()
            print(f"Grid(s) 0-{self.ptIndex} Generated succesfully")
            print(f"Grid(s) 0-{self.ptIndex-1} Displayed and analyzed succesfully")

            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + self.currentGrid.zero[0], bound + self.currentGrid.zero[0])
            self.ax.set_ylim(-bound + self.currentGrid.zero[1], bound + self.currentGrid.zero[1])
            
            self.saveAndExit()
        else:
            os.makedirs(self.localTrashPath)
            os.makedirs(f'{self.localTrashPath}ByTimeStep/')
            self.currentGrid.genTilingVerts()
            print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            print('Tile neighbourhood generated')
            self.currentGrid.genTiling()
            self.tilingIter(self.currentGrid)

    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridList[self.ptIndex-2].fig.clf()
        #    self.multigridList[self.ptIndex-2].ax.cla()

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
            self.currentGrid.genTilingVerts()
            print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            print('Tile neighbourhood generated')
            self.currentGrid.genTiling()
            print('Original tile populated')
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
            self.colValDicts.append(origGrid.colValDict)

            if self.isBoundaried:
                self.numBoundaries.append(nextGrid.numBoundaries)
                self.numUnstable.append(len(origGrid.unstableTiles))
                if self.ptIndex==1:
                    self.origNumBoundaries = nextGrid.numBoundaries
            #if self.isBoundaried and (self.ptIndex > 1):
                #if nextGrid.numBoundaries/self.origNumBoundaries == self.prevPercentStable:
                    #self.percentStableRepeatCount += 1
                #else:
                    #self.percentStableRepeatCount = 0
                #self.prevPercentStable = nextGrid.numBoundaries/self.origNumBoundaries
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
        sM = ''
        if self.shiftProp[0]:
            sM = 'zeroes'
        elif self.shiftProp[1]:
            sM = 'random'
        elif self.shiftProp[2]:
            sM = 'halves'
        title = f'dim={self.dim}, size={self.size}, sC={self.sC}, sM={sM}, gen={self.ptIndex}'
        self.ax.set_title(title)
        print(f'Grid {self.ptIndex} complete', end='\r')
        self.ptIndex += 1
        return axis

    ############################
    ## Parameter Safety Check ##
    ############################
    def parameterSafetyCheck(self):
        print(' '*50)
        flawCaught, critFlawCaught = False, False
        ## This ensures that all multigrid parameters are valid
        if self.dim<3:
            print(f"   --Dimmension '{self.dim}' < 3, dimmensions smaller than three cannot be projected")
            print('     Dimmension defaulted to 5')
            self.dim=3
            flawCaught, critFlawCaught = True, True
        if self.size<0:
            print(f"   --Size '{self.size}' < 0, sizes the minimum size of a tiling is 0")
            print('     Size defaulted to 0')
            self.size=0
            flawCaught, critFlawCaught = True, True
        if self.tileSize<0:
            print(f"  --Tiling size '{self.tileSize}' < 0, the size of each tile must be positive, as do all lengths")
            print('     Tiling Size deafulted to 10')
            self.tileSize=10
            flawCaught, critFlawCaught = True, True
        if self.numColors<1:
            print(f"   --The number of colors '{self.numColors}' < 1, the number of colors must be greater than 0")
            print('     The number of colors defaulted to 10')
            if self.numStates>=10:
                self.numColors=10
            else:
                self.numColors=self.numStates
            flawCaught, critFlawCaught = True, True
        if self.numStates<1:
            print(f"   --The number of states '{self.numStates}' < 1, the number of states must be greater than 0")
            print(f"     The number of states defaulted to the number of states: '{self.numStates}'")
            self.numStates=self.numColors
            flawCaught, critFlawCaught = True, True
        if self.maxGen<0:
            print(f"   --The maximum generation '{self.maxGen}' < 0, the maximum generation must be greater than or equal to 0")
            print('     The maximum generation defaulted to 0')
            self.maxGen=0
            flawCaught, critFlawCaught = True, True
        if self.gol==True and self.isBoundaried==True:
            print('   --The gamemode is set to GOL, yet isBoundaries=True')
            print('     isBoudnaried defaulted to False')
            self.isBoundaried=False
            flawCaught, critFlawCaught = True, True
        if self.initialValue[1] and self.dim<7:
            print(f"        --The initial value condition of the space is a function of dim, which in this case is '{self.dim}' < 7")
            print('          In order to increase complexity, try increasing dim to 7 or more (preferably 15+)')
            flawCaught = True
        if self.initialValue[2] and self.size<5:
            print(f"        --The initial value condition of the space is a function of size, which in this case is '{self.size}' < 5")
            print('          Please keep in mind that part of the space may not be playable (ie the background).  In order to imcrease complexity,')
            print('          try increasing size to 5 or more (note that size is more than twice as costly as dimmension)')
            flawCaught = True
        if self.initialValue[3] and self.dim<11:
            print('         --The initial value condition of the space is a function of the tile type, itself a function of dim,')
            print(f"          which in this case is '{self.dim}' < 11.")
            print('          Please keep in mind that tileType ~dim/2. In order to increase complexity, try increasing dim to 11 or more')
            flawCaught = True
        if self.initialValue[1]:
            print('-valuedByDim detected, numColors defaulted to dim')
            self.numColors = self.dim
            flawCaught, critFlawCaught = True, True
        elif self.initialValue[2]:
            print('-valuedBySize detected, numColors defaulted to size+1')
            self.numColors = self.size+1
            flawCaught, critFlawCaught = True, True
        elif self.initialValue[3]:
            print('-valuedByTileType detected, numColors and numStates defaulted to the number of types of tiles in the grid')
            if self.dim%2 == 0:
                self.numStates = int((self.dim/2)-1)
            else:
                self.numStates = int((self.dim-1)/2)
            self.numColors = self.numStates
            flawCaught, critFlawCaught = True, True
            print(f' In this case, numColors and numStates both equal {self.numColors}')
        if self.numColors>self.numStates:
            print(f"   --The number of colors '{self.numColors}' > the numebr of states '{self.numStates}', this is impossible")
            print(f"     The number of colors and the number of states both defaulted to numColors('{self.numColors}')")
            self.numStates = self.numColors
            flawCaught, critFlawCaught = True, True
        print('-'*50)
        print('All Paramters Validated')
        print('Parameter Safety Check Passed')
        print(' ')
        if flawCaught:
            print('     Flaw(s) were caught on the input condition, no flaws correction implemented, attempting build')
        if flawCaught and critFlawCaught:
            print(' ')
        if critFlawCaught:
            print('     Critical flaw(s) were caught on the input condition, flaw correction implemented, attempting build')
        if not flawCaught and not critFlawCaught:
            print('     No flaws were caught on the input condition, parameterization successful, attempting build')
        print('-'*50)

    #######################
    ## Generator Methods ##
    #######################
    def genFrames(self):
        while (self.ptIndex < self.maxGen)  and self.continueAnimation:
            self.ax.cla()
            yield self.ptIndex
        yield StopIteration

    def genColors(self):
        if self.manualCols or self.numColors>19:
            ## Manually create colors
            # (hue, saturation, value)
            hsvCols = [(x/self.numColors, 1, 0.75) for x in range(self.numColors)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvCols))
        else:
            ## Classic colors
            self.colors = sns.color_palette("bright", self.numColors)
            #self.colors = sns.color_palette("husl", self.numColors)
            #self.colors = sns.color_palette("cubehelix", self.numColors)

            ## Divergent colors
            #self.colors = sns.color_palette("BrBG", self.numColors)
            #self.colors = sns.color_palette("coolwarm", self.numColors)

            ## Gradient colors
            #self.colors = sns.cubehelix_palette(self.numColors, dark=0.1, light=0.9)

    def genDirectoryPaths(self):
        ## rootPath and path data
        filePath = os.path.realpath(__file__)
        self.rootPath = filePath.replace('src/MultigridList.py', 'outputData/fitMultigridData/')
        self.multigridListInd = str(rd.randrange(0, 2000000000))
        self.gridPath = f'listInd{self.multigridListInd}/'
        ## localPaths
        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('fitMultigridData', 'unfitMultigridData')
        ## gifPaths
        self.gifPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}Animation.gif'
        self.gifTrashPath = self.gifPath.replace('fitMultigridData', 'unfitMultigridData')
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
        ## avgGenDiffs
        self.genAvgChangePath, self.genAvgChangeTrashPath = self.genPngPaths('genAvgChange')
        ## updatedStabilityPath
        self.updatedStabilityPath, self.updatedStabilityTrashPath = self.genPngPaths('updatedStabilityPath')
        ## jsonPath
        self.detailedInfoPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}detaileInfo.json'
        self.detailedInfoTrashPath = self.detailedInfoPath.replace('fitMultigridData', 'unfitMultigridData')

    def genPngPaths(self, pngName):
        pngPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}{pngName}.png'
        return pngPath, pngPath.replace('fitMultigridData', 'unfitMultigridData')


    def genBounds(self):
        sampleDef = 1000
        upper = 1
        if self.boundaryReMap:
            upper = 4
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
                #self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
                if rd.randrange(0, upper) == 0:
                    self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]
                    self.boundToPC[bound] = 0
                else:
                    self.boundToKeyCol[bound] = self.colors[i]
                    self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef

    ############################
    ## Save Tiling Statistics ##
    ############################
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
        boundaryFig.update_layout(title=f'Boundary Stats vs. Tile Generation             totalNumTiles @{self.numTilesInGrid}',
                                  xaxis_title='tile Index',
                                  yaxis_title='number of tiles/boundaries')
        boundaryFig.write_image(self.boundaryTrashPath)
    def saveValFig(self):
        valFig = go.Figure()
        x = [str(i) for i in range(len(self.valAvgs))]
        valFig.add_trace(go.Scatter(x=x, y=self.valAvgs, name='grid value average', line=dict(color='firebrick', width=4)))
        valFig.add_trace(go.Scatter(x=x, y=self.valStdDevs, name='grid value std. dev.', line=dict(color='royalblue', width=4)))
        valFig.update_layout(title=f'Value Stats vs. Tile Generation             numStates @{self.numStates}',
                                  xaxis_title='tile Index',
                                  yaxis_title='statistic value')
        valFig.write_image(self.valTrashPath)
    def saveColorCompFig(self):
        avgGens = self.genColAvg()
        colCompFig = go.Figure()
        x = [str(i) for i in range(len(self.colValDicts))]
        numBorderTiles = [(len(self.currentGrid.specialTiles)-len(self.invalidSet)) for i in range(len(self.colValDicts))]
        self.colComp = [[] for _ in range(len(self.colors))]
        for tilingInd in range(len(self.colValDicts)):
            for i, color in enumerate(self.colors):
                color = tuple(color) if type(color) is list else color
                if color in self.colValDicts[tilingInd]:
                    self.colComp[i].append(self.colValDicts[tilingInd].get(color))
                else:
                    self.colComp[i].append(0)
        colCompFig.add_trace(go.Scatter(x=x, y=numBorderTiles, name='num border tiles', line=dict(color='black', width=4)))
        for i, invalidSet in enumerate(self.invalidSets):
            invalidSetTiles = [len(invalidSet) for x in range(len(self.colValDicts))]
            colCompFig.add_trace(go.Scatter(x=x, y=invalidSetTiles, name=f'invalid set {i}', line=dict(color=f'{self.invalidColors[i]}', width=4)))
        colCompFig.add_trace(go.Scatter(x=x, y=avgGens, name='color avg', line=dict(color=f'red', width=4)))
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
        normAvgGens = [genAvg/tilesPerGen[tilingInd] for tilingInd, genAvg in enumerate(self.genColAvg())]
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
        normColCompFig.add_trace(go.Scatter(x=x, y=totalNumTiles, name='total percentage', line=dict(color='black', width=4)))
        normColCompFig.add_trace(go.Scatter(x=x, y=normAvgGens, name='normalized color avg', line=dict(color='red', width=4)))
        for color, colHist, i in zip(self.colors, normColComp, itertools.count()):
            color = tuple(color) if type(color) is list else color
            normColCompFig.add_trace(go.Scatter(x=x, y=colHist, name=f'color{i}', line=dict(color=f'rgb{color}', width=4)))
        normColCompFig.update_layout(title='Normalized Color Composition vs. Tile Generation',
                                 xaxis_title='tile index',
                                 yaxis_title='percent composition of evaluated tiles')
        normColCompFig.write_image(self.normColCompTrashPath)
    def saveAvgGenDiffFig(self):
        avgGenDiffs = self.genColGenDiffs()

        avgGenDiffsFig = go.Figure()
        avgGenDiffx = [str(i) for i in range(len(self.colValDicts)-1)]
        avgGenDiffsFig.add_trace(go.Scatter(x=avgGenDiffx, y=avgGenDiffs, name='avgColDiff', line=dict(color='black', width=4)))
        avgGenDiffsFig.update_layout(title='Derivative of average color composition',
                                     xaxis_title='tile index',
                                     yaxis_title='average change')
        avgGenDiffsFig.write_image(self.genAvgChangeTrashPath)
    def saveTilingInfo(self):
        tileData = {'dim':self.dim, 'size':self.size, 'sC':self.sC, 'shiftProp':self.shiftProp,
                    'shiftVect':self.shiftVect,
                    'numTilesInGrid':self.numTilesInGrid,
                    'minGen':self.minGen, 'maxGen':self.maxGen, 'fitGen':self.fitGen, 'printGen':self.printGen,
                    'numColors':self.numColors, 'numStates':self.numStates, 'manualCols':self.manualCols,
                    'isValued':self.isValued, 'initialValue':self.initialValue, 'valRatio':self.valRatio,
                    'isBoundaried':self.isBoundaried, 'boundaryReMap':self.boundaryReMap, 'boundaryApprox':self.boundaryApprox,
                    'gol':self.gol, 'tileOutline':self.tileOutline, 'alpha':self.alpha, 'overide':self.overide,
                    'boundToPC':self.boundToPC,
                    'boundToCol':self.boundToCol, 'boundToKeyCol':self.boundToKeyCol, 'colors':self.colors
                   }
        with open(self.detailedInfoTrashPath, 'w') as file:
            json.dump(tileData, file, indent=4)
            file.close()
        #readTilingInfo(self.detailedInfoTrashPath)
    def genColAvg(self):
        avgGens = []
        for tilingInd in range(len(self.colValDicts)):
            totalAvgGen = 0
            for color in self.colors:
                color = tuple(color) if type(color) is list else color
                if color in self.colValDicts[tilingInd]:
                    totalAvgGen += self.colValDicts[tilingInd].get(color)
                else:
                    totalAvgGen += 0
            avgGen = totalAvgGen / (len(self.colors)-1)
            avgGens.append(avgGen)
        return avgGens
    ## The derivative of the average color composition
    def genColGenDiffs(self):
        avgGenDiffs = []
        for tilingInd in range(1, len(self.colValDicts)):
            totalGenDiff = 0
            for color in self.colors:
                color = tuple(color) if type(color) is list else color
                if color in self.colValDicts[tilingInd] and color in self.colValDicts[tilingInd-1]:
                    totalGenDiff += -(self.colValDicts[tilingInd].get(color) - self.colValDicts[tilingInd-1].get(color))
                elif color in self.colValDicts[tilingInd-1]:
                    totalGenDiff += self.colValDicts[tilingInd-1].get(color)
                else:
                    totalGenDiff += 0
            avgGenDiff = totalGenDiff / (len(self.colors)-1)
            avgGenDiffs.append(avgGenDiff)
        return avgGenDiffs

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
        self.saveAvgGenDiffFig()
        ## Relocate fit tilings
        if self.ptIndex > self.fitGen:
            files = glob.glob(self.localTrashPath)
            for f in files:
                shutil.move(f, self.localPath)
        ## Print execution completion and time
        print('-'*50)
        print(' '*50)
        print('-'*50)
        print('Algorithm Completed')
        print('Executed in {} seconds'.format(time.time()-self.startTime))
        print('-'*50)




    #####################################################
    ## Old Methods For Generating Animation By Picture ##
    #####################################################
    def genNextGrid(self, origGrid):
        nextGrid = self.currentGrid.genNextValuedGridState()
        self.ptIndex += 1
        self.currentGrid = nextGrid
        return nextGrid
                        
    def tilingIter(self, rootPop):
        rootPop.displayTiling(animating=False)
    ######################################################
    ######################################################
    ######################################################




def readTilingInfo(path):
    with open(path, 'r') as fp:
        tilingInfo = json.load(fp)
    return tilingInfo



def cleanFileSpace(cleaning, fitClean=False, unfitClean=False):
    filePath = os.path.realpath(__file__)
    rootPath = filePath.replace('src/MultigridList.py', '/outputData/')
    if cleaning:
        if fitClean:
            files = glob.glob(f'{rootPath}fitMultigridData/*')
            for f in files:
                shutil.rmtree(f)
        if unfitClean:
            tFiles = glob.glob(f'{rootPath}unfitMultigridData/*')
            for tF in tFiles:
                shutil.rmtree(tF)


###############################
## Local Animation Execution ##
###############################
## Main definition
def main():
    #p = cProfile.Profile()
    # Enable profiling
    #p.enable()
    
    cleanFileSpace(True, fitClean=False, unfitClean=True)

    dim = 5
    size = 5
    sC = 0
    
    manualCols = True
    tileOutline = True
    alpha = 1
    
    ## You can use this to overide any non properly tiled tilings
    shiftVect = None
    shiftZeroes, shiftRandom, shiftByHalves = True, False, False
    shiftProp = (shiftZeroes, shiftRandom, shiftByHalves)
    
    numColors = 20
    numStates = 10000
    boundaryReMap = True

    ## valuedRandomly, valuedByDim, valuedBySize, valuedByTT
    initialValue = (True, False, False, False)

    minGen = 20
    maxGen = 20
    fitGen = 21

    isBoundaried = True
    ## Setting boundary approx trades time complexity for calculating the exact tiling
    ## Setting boundaryApprox as True improves time complexity and gives tiling approximation
    boundaryApprox = False

    ## Change gamemode to GOL
    gol = False

    ## Overide ensures maxGen generations
    overide = False

    printGen = 0

    borderSet = {0, 1, 2, 3, 4, 5, 6}
    
    invalidSets = []

    invalidSet = set()
    invalidSet2 = set()
    # for a in range(-size, size+1):
    #     for b in range(-size, size+1):
    #         invalidSet.add((0, a, 1, b))
    for r in range(dim):
        for s in range(r+1, dim):
            invalidSet.add((r, 0, s, 0))
            invalidSet2.add((r, 2, s, 2))
            
    invalidSet.update([
        # (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 2), (0, 2, 1, 0)
        ])
    invalidSets.append(invalidSet)
    invalidSets.append(invalidSet2)
    borderColor = 'black'
    invalidColors = ['black', 'black']
    borderVal = numStates
    invalidVals = [numStates, numStates]
    dispBorder = True
    dispInvalid = True


    numIterations = 1
    for _ in range(numIterations):
        MultigridList(dim, size, shiftVect, sC=sC, shiftProp=shiftProp,
                      minGen=minGen, maxGen=maxGen, fitGen=fitGen, printGen=printGen,
                      numColors=numColors, manualCols=manualCols, numStates=numStates,
                      isValued=True, initialValue=initialValue, valRatio=0.5,
                      isBoundaried=isBoundaried, boundaryReMap=boundaryReMap, boundaryApprox=boundaryApprox,
                      gol=gol, tileOutline=tileOutline, alpha=alpha, overide=overide,
                      borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
                      borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid)

    #p.disable()
    #p.print_stats()
    # Dump the stats to a file
    #p.dump_stats("results.prof")

## Local main call
if __name__ == '__main__':
    main()



## TODO: If dim is eve, each iteration over multigrid has two not one invalid tilings, thats why we get some lines and not rhombs i think
## TODO: Fix shiftVect
## TODO: Fix tile type