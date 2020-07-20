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


## MultigridList Class
class MultigridList:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, size,
                 sC=0, sV=None, sP=(False,True,True),
                 isValued=True, valRatio=None,
                 numColors=10, numStates=100, boundaryReMap=False,
                 colors=None, initialValue=(True,False,False,False),
                 bounds=None, boundToCol=None, boundToPC=None,
                 minGen=8, maxGen=10, fitGen=10, printGen=0,
                 isBoundaried=False, boundaryApprox=False,
                 gol=False, overide=False,
                 manualCols=True, tileOutline=False, alpha=1,
                 borderSet={0,1,2,3,4,5,6}, borderColor='black', borderVal=-1, dispBorder=False,
                 invalidSets=[], invalidColors=[], invalidVals=[], dispInvalid=True,
                 iterationNum=0, captureStatistics=False,
                 ga=False, gaPath=None):
        
        ## Animation on
        self.animating = True
        ## Early animation exit param
        self.continueAnimation = True
        ## Constant grid parameters
        self.startTime = time.time()


        ## Main Imports ##
        self.dim, self.size = dim, size
        self.sC, self.sV, self.sP = sC, sV, sP
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = sP
        
        ## State value settings
        self.isValued, self.valRatio = isValued, valRatio
        
        ## States and colors
        self.numColors, self.numStates, self.boundaryReMap = numColors, numStates, boundaryReMap
        
        ## Initial condition population method for tile values
        self.initialValue = initialValue
        
        ## Choose a color scheme
        self.manualCols = manualCols
        if colors == None:
            self.genColors()
        else:
            self.colors = colors
        
        ## Generate ruleset
        if bounds == None:
            self.genBounds()
        else:
            self.bounds, self.boundToCol, self.boundToPC = bounds, boundToCol, boundToPC

        ## Generation counters
        self.minGen, self.maxGen, self.fitGen, self.printGen = minGen, maxGen, fitGen, printGen

        ## Boundary variables
        self.isBoundaried, self.boundaryApprox = isBoundaried, boundaryApprox

        ## Game of life gamemode
        self.gol = gol
        self.overide = overide
        self.tileOutline, self.alpha, = tileOutline, alpha
        
        self.borderSet, self.borderColor, self.borderVal, self.dispBorder = borderSet, borderColor, borderVal, dispBorder
        self.invalidSets, self.invalidColors, self.invalidVals, self.dispInvalid = invalidSets, invalidColors, invalidVals, dispInvalid
        
        self.captureStatistics = captureStatistics
        self.silence = True if ga else False
        self.ga = ga
        
        ## Do not mess with this
        self.tileSize = 10
        
        ## Generate directory names and paths
        self.gaPath = gaPath
        self.genDirectoryPaths()
        os.makedirs(self.localTrashPath)
        
        ## Parameter Safety Check
        if iterationNum==0:
            self.parameterSafetyCheck()
            
        ## Independent Actions of Constructor ##
        self.invalidSet = set()
        for invalidSet in self.invalidSets:
            self.invalidSet.update(invalidSet)
        
        ## The number of tiles in this grid
        self.numTilesInGrid = (math.factorial(self.dim)/(2*(math.factorial(self.dim-2))))*((2*self.size+1)**2)

        ## Keep track of stability
        self.percentStableRepeatCount = 0
        self.prevPercentStable = 1
        
        
        ## Generate lists used to keep track of statistics
        if self.animating:
            if self.isBoundaried:
                self.numBoundaries, self.numUnstable = [], []
                
            if self.captureStatistics:
                self.numChanged = []
                self.percentStables, self.percentUnstables, self.percentTotals = [], [], []
                self.normPercentStables, self.normPercentUnstables = [], []
                self.numVals, self.valAvgs, self.valStdDevs = [], [], []
                self.colValDicts = []

        ## Current tile count
        self.ptIndex = 0

        ## Create and animate original tiling
        self.currentGrid = Multigrid(self.dim, self.size, shiftVect=self.sV, sC=self.sC, shiftProp=sP,
                                     numTilesInGrid=self.numTilesInGrid,
                                     startTime=self.startTime, rootPath=self.localTrashPath, ptIndex=self.ptIndex, 
                                     numStates=self.numStates, colors=self.colors,
                                     isValued=self.isValued, initialValue=self.initialValue,  valRatio=self.valRatio,
                                     boundToCol=self.boundToCol, boundToPC=self.boundToPC,
                                     isBoundaried=isBoundaried, bounds=self.bounds, boundaryApprox=self.boundaryApprox,
                                     gol=self.gol, tileOutline=self.tileOutline, alpha=self.alpha, printGen=self.printGen,
                                     borderSet=self.borderSet, invalidSets=self.invalidSets, borderColor=self.borderColor, invalidColors=self.invalidColors,
                                     borderVal=self.borderVal, invalidVals=self.invalidVals, dispBorder=self.dispBorder, dispInvalid=self.dispInvalid,
                                     captureStatistics=self.captureStatistics)
        self.currentGrid.invalidSet = self.invalidSet

        if self.animating:
            self.animatingFigure = plt.figure()
            self.ax = plt.axes()
            self.ax.axis('equal')

            self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False)
            self.updateDir()
            self.saveTilingInfo()
            if not self.silence:
                print(f"Grid(s) 0-{self.ptIndex} Generated succesfully")
                print(f"Grid(s) 1-{self.ptIndex} Displayed and analyzed succesfully")

            lim = 10
            bound = lim*(self.size+self.dim-1)**1.2
            self.ax.set_xlim(-bound + self.currentGrid.zero[0], bound + self.currentGrid.zero[0])
            self.ax.set_ylim(-bound + self.currentGrid.zero[1], bound + self.currentGrid.zero[1])
            
            self.saveAndExit()
        else:
            os.makedirs(self.localTrashPath)
            os.makedirs(f'{self.localTrashPath}ByTimeStep/')
            
            self.currentGrid.genTilingVerts()
            if not self.silence:
                print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            if not self.silence:
                print('Tile neighbourhoods generated')
            self.currentGrid.genTiling()
            if not self.silence:
                print('Initial tiling populated')
            self.tilingIter(self.currentGrid)

    def initPlot(self):
        self.ax.cla()
        #if self.numTilings > 1:
        #    self.multigridList[self.ptIndex-2].fig.clf()
        #    self.multigridList[self.ptIndex-2].ax.cla()

    ########################################
    ## Update Methods For Animation Cycle ##
    ########################################
    def updateDir(self):
        self.anim.save(self.gifTrashPath, writer=PillowWriter(fps=4))

    def updateAnimation(self, i):
        if self.ptIndex==0:
            self.currentGrid.genTilingVerts()
            if not self.silence:
                print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            if not self.silence:
                print('Tile neighbourhood generated')
            self.currentGrid.genTiling()
            if self.captureStatistics:
                self.numNotEvaluated = self.currentGrid.borderSetLen + len(self.invalidSet)
            if not self.silence:
                print('Original tile populated')
            if self.printGen == 0:
                axis = self.currentGrid.displayTiling()
                self.origAx = axis
            else:
                axis = None
        else:
            origGrid = self.currentGrid
            origGrid.ax.cla()
            nextGrid, axis = origGrid.genNextValuedGridState(animating=True, boundaried=self.isBoundaried)

            if self.isBoundaried:
                self.numBoundaries.append(nextGrid.numBoundaries)
                self.numUnstable.append(len(origGrid.unstableTiles))
                if self.ptIndex==1:
                    self.origNumBoundaries = nextGrid.numBoundaries
            if self.captureStatistics:      
                self.percentStables.append(origGrid.percentStable)
                self.percentUnstables.append(origGrid.percentUnstable)
                self.percentTotals.append(origGrid.totalPercent)
                self.normPercentStables.append(origGrid.normPercentStable)
                self.normPercentUnstables.append(origGrid.normPercentUnstable)
                self.valAvgs.append(origGrid.valAvg)
                self.numVals.append(len(origGrid.values))
                self.valStdDevs.append(origGrid.valStdDev)
                self.colValDicts.append(origGrid.colValDict)
                self.numChanged.append(origGrid.numChanged)
            #if self.isBoundaried and (self.ptIndex > 1):
                #if nextGrid.numBoundaries/self.origNumBoundaries == self.prevPercentStable:
                    #self.percentStableRepeatCount += 1
                #else:
                    #self.percentStableRepeatCount = 0
                #self.prevPercentStable = nextGrid.numBoundaries/self.origNumBoundaries
                #print(nextGrid.numBoundaries/self.origNumBoundaries, origGrid.numBoundaries, self.percentStableRepeatCount)
            
            ## Stop the animation before maxGen
            if self.ptIndex < self.minGen or self.overide:
                pass
            elif self.isBoundaried and (not self.gol) and (self.ptIndex > 1) and ((nextGrid.numBoundaries/self.origNumBoundaries < 0.5) or (self.percentStableRepeatCount > 1)):
                self.continueAnimation = False
            self.currentGrid = nextGrid
            ## I think this helps with the memory leaks
            del origGrid
            
        ## Format animation title
        sM = ''
        if self.sP[0]:
            sM = 'zeroes'
        elif self.sP[1]:
            sM = 'random'
        elif self.sP[2]:
            sM = 'halves'
        title = f'dim={self.dim}, size={self.size}, sC={self.sC}, sM={sM}, gen={self.ptIndex}'
        self.ax.set_title(title)
        if not self.silence:
            print(f'Grid {self.ptIndex} complete', end='\r')
        self.ptIndex += 1
        return axis  

    ############################
    ## Parameter Safety Check ##
    ############################
    def parameterSafetyCheck(self):
        flaws = {}
        if not self.silence:
            print(' '*50)
        flawCaught, critFlawCaught = False, False
        flawAmt, critFlawAmt = 0, 0
        ## This ensures that all multigrid parameters are valid
        if self.dim<3:
            if not self.silence:
                print(f"   --Dimmension '{self.dim}' < 3, dimmensions smaller than three cannot be projected")
                print('     Dimmension defaulted to 5')
            flaws.update({'f0':True})
            self.dim=3
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f0':False})
            
        if self.size<0:
            if not self.silence:
                print(f"   --Size '{self.size}' < 0, sizes the minimum size of a tiling is 0")
                print('     Size defaulted to 0')
            flaws.update({'f1':True})
            self.size=0
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f1':False})
            
        if self.tileSize<0:
            if not self.silence:
                print(f"  --Tiling size '{self.tileSize}' < 0, the size of each tile must be positive, as do all lengths")
                print('     Tiling Size deafulted to 10')
            flaws.update({'f2':True})
            self.tileSize=10
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f2':False})
            
        if self.numColors<1:
            if not self.silence:
                print(f"   --The number of colors '{self.numColors}' < 1, the number of colors must be greater than 0")
                print('     The number of colors defaulted to 10')
            flaws.update({'f3':True})
            if self.numStates>=10:
                self.numColors=10
            else:
                self.numColors=self.numStates
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f3':False})
            
        if self.numStates<1:
            if not self.silence:
                print(f"   --The number of states '{self.numStates}' < 1, the number of states must be greater than 0")
                print(f"     The number of states defaulted to the number of states: '{self.numStates}'")
            flaws.update({'f4':True})
            self.numStates=self.numColors
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f4':False})
            
        if self.maxGen<0:
            if not self.silence:
                print(f"   --The maximum generation '{self.maxGen}' < 0, the maximum generation must be greater than or equal to 0")
                print('     The maximum generation defaulted to 0')
            flaws.update({'f5':True})
            self.maxGen=0
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f5':False})
            
        if self.gol==True and self.isBoundaried==True:
            if not self.silence:
                print('   --The gamemode is set to GOL, yet isBoundaries=True')
                print('     isBoudnaried defaulted to False')
            flaws.update({'f6':True})
            self.isBoundaried=False
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f6':False})
            
        if self.initialValue[1] and self.dim<7:
            if not self.silence:
                print(f"        --The initial value condition of the space is a function of dim, which in this case is '{self.dim}' < 7")
                print('          In order to increase complexity, try increasing dim to 7 or more (preferably 15+)')
            flaws.update({'f7':True})
            flawCaught = True
            flawAmt += 1
        else:
            flaws.update({'f7':False})
            
        if self.initialValue[2] and self.size<5:
            if not self.silence:
                print(f"        --The initial value condition of the space is a function of size, which in this case is '{self.size}' < 5")
                print('          Please keep in mind that part of the space may not be playable (ie the background).  In order to imcrease complexity,')
                print('          try increasing size to 5 or more (note that size is more than twice as costly as dimmension)')
            flaws.update({'f8':True})
            flawCaught = True
            flawAmt += 1
        else:
            flaws.update({'f8':False})
            
        if self.initialValue[3] and self.dim<11:
            if not self.silence:
                print('         --The initial value condition of the space is a function of the tile type, itself a function of dim,')
                print(f"          which in this case is '{self.dim}' < 11.")
                print('          Please keep in mind that tileType ~dim/2. In order to increase complexity, try increasing dim to 11 or more')
            flaws.update({'f9':True})
            flawCaught = True
            flawAmt += 1
        else:
            flaws.update({'f9':False})
            
        if self.initialValue[1]:
            if not self.silence:
                print('-valuedByDim detected, numColors defaulted to dim')
            flaws.update({'f10':True})
            self.numColors = self.dim
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        elif self.initialValue[2]:
            if not self.silence:
                print('-valuedBySize detected, numColors defaulted to size+1')
            flaws.update({'f10':True})
            self.numColors = self.size+1
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        elif self.initialValue[3]:
            if not self.silence:
                print('-valuedByTileType detected, numColors and numStates defaulted to the number of types of tiles in the grid')
            flaws.update({'f10':True})
            if self.dim%2 == 0:
                self.numStates = int((self.dim/2)-1)
            else:
                self.numStates = int((self.dim-1)/2)
            self.numColors = self.numStates
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
            if not self.silence:
                print(f' In this case, numColors and numStates both equal {self.numColors}')
        else:
            flaws.update({'f10':False})
            
        if self.numColors>self.numStates:
            if not self.silence:
                print(f"   --The number of colors '{self.numColors}' > the numebr of states '{self.numStates}', this is impossible")
                print(f"     The number of colors and the number of states both defaulted to numColors('{self.numColors}')")
            flaws.update({'f11':True})
            self.numStates = self.numColors
            flawCaught, critFlawCaught = True, True
            flawAmt += 1
            critFlawAmt += 1
        else:
            flaws.update({'f11':False})
            
        if not self.silence:
            print('-'*50)
            print('All Paramters Validated')
            print('Parameter Safety Check Passed')
            print(' ')
        if flawCaught:
            flaw = f'   {flawAmt}  Flaw(s) were caught on the input condition, no flaws correction implemented, attempting build'
            if self.silence:
                flaws.update({'out':flaw})
            else:
                print(flaw)
        if not self.silence and flawCaught and critFlawCaught:
            print(' ')
        if critFlawCaught:
            critFlaws = f'   {critFlawAmt}  Critical flaw(s) were caught on the input condition, flaw correction implemented, attempting build'
            if self.silence:
                flaws.update({'out':critFlaws})
            else:
                print(critFlaws)
        if not flawCaught and not critFlawCaught:
            noFlaws = '     No flaws were caught on the input condition, parameterization successful, attempting build'
            if self.silence:
                flaws.update({'out':noFlaws})
            else:
                print(noFlaws)
        if not self.silence:
            print('-'*50)
        with open(self.parameterSafetyCheckTrashPath, 'w') as fw:
            json.dump(flaws, fw, indent=4)
            fw.close()

    #######################
    ## Generator Methods ##
    #######################
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
                if rd.randrange(0, upper) == 3:
                    self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]
                    self.boundToPC[bound] = 0
                else:
                    self.boundToKeyCol[bound] = self.colors[i]
                    self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef

    def genFrames(self):
        while (self.ptIndex < self.maxGen)  and self.continueAnimation:
            self.ax.cla()
            yield self.ptIndex
        yield StopIteration

    def genDirectoryPaths(self):
        if self.gaPath == None:
            ## rootPath and path data
            filePath = os.path.realpath(__file__)
            self.rootPath = filePath.replace('src/MultigridList.py', 'outputData/fitMultigridData/')
        else:
            self.rootPath = self.gaPath
        if not self.gaPath == None and self.printGen == 0:
            self.multigridListInd = 'fitDisp'
            self.gridPath = 'fitDisp/'
        else:
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
        self.updatedStabilityPath, self.updatedStabilityTrashPath = self.genPngPaths('updatedStability')
        ## fitnessesPath
        self.fitnessesPath, self.fitnessesTrashPath = self.genPngPaths('fitnesses')
        ## jsonPaths
        self.detailedInfoPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}detaileInfo.json'
        self.detailedInfoTrashPath = self.detailedInfoPath.replace('fitMultigridData', 'unfitMultigridData')
        self.parameterSafetyCheckPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}parameterSafetyCheck.json'
        self.parameterSafetyCheckTrashPath = self.parameterSafetyCheckPath.replace('fitMultigridData', 'unfitMultigridData')
        

    def genPngPaths(self, pngName):
        pngPath = f'{self.localPath}listInd{self.multigridListInd[0:3]}{pngName}.png'
        return pngPath, pngPath.replace('fitMultigridData', 'unfitMultigridData')
    
                

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
        boundaryFig.update_layout(title=f'Boundary Stats vs. Tile Generation             totalNumTiles: {self.numTilesInGrid}',
                                  xaxis_title='tile Index',
                                  yaxis_title='number of tiles/boundaries')
        boundaryFig.write_image(self.boundaryTrashPath)
    def saveValFig(self):
        valFig = go.Figure()
        x = [str(i) for i in range(len(self.valAvgs))]
        valFig.add_trace(go.Scatter(x=x, y=self.valAvgs, name='grid value average', line=dict(color='firebrick', width=4)))
        valFig.add_trace(go.Scatter(x=x, y=self.valStdDevs, name='grid value std. dev.', line=dict(color='royalblue', width=4)))
        valFig.update_layout(title=f'Value Stats vs. Tile Generation             numStates: {self.numStates}',
                                  xaxis_title='tile Index',
                                  yaxis_title='statistic value')
        valFig.write_image(self.valTrashPath)
    def saveColorCompFig(self):
        #self.avgGens = self.genColAvg()
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
        self.avgGenDiffs = [(self.valAvgs[i]-self.valAvgs[i+1]) for i in range(len(self.valAvgs)-1)]
        avgGenDiffsFig = go.Figure()
        avgGenDiffx = [str(i) for i in range(len(self.colValDicts)-1)]
        avgGenDiffsFig.add_trace(go.Scatter(x=avgGenDiffx, y=self.avgGenDiffs, name='avgColDiff', line=dict(color='black', width=4)))
        avgGenDiffsFig.update_layout(title='Derivative of average color composition',
                                     xaxis_title='tile index',
                                     yaxis_title='number of tiles whose state changed')
        avgGenDiffsFig.write_image(self.genAvgChangeTrashPath)
    def saveFitnessFig(self):
        self.maxNumEvaluated = self.numTilesInGrid - self.numNotEvaluated
        # for valAvg, numChanged in zip(self.valAvgs, self.numChanged):
        #     print((valAvg/self.numStates), max(numChanged,0)/self.maxNumEvaluated)
        fitnesses = [(valAvg/self.numStates)*(max(numChanged, 0)/self.maxNumEvaluated) for valAvg, numChanged in zip(self.valAvgs, self.numChanged)]
        fitnessesFig = go.Figure()
        x = [str(i) for i in range(len(fitnesses))]
        self.tilingFitness = sum(fitnesses)
        self.avgFit = self.tilingFitness/len(fitnesses)
        fitnessesFig.add_trace(go.Scatter(x=x, y=fitnesses, name=f'tiling fitness', line=dict(color='black', width=4)))
        fitnessesFig.update_layout(title=f'Tiling Fitness vs. Tile Generation,     totalFit: {str(self.tilingFitness)[:5]},   avgFit: {str(self.avgFit)[:5]}',
                                     xaxis_title='tile index',
                                     yaxis_title='fitness value')
        fitnessesFig.write_image(self.fitnessesTrashPath)
    def saveTilingInfo(self):
        invalidSetsRep = [list(invalidSet) for invalidSet in self.invalidSets]
        tileData = {'dim':self.dim, 'size':self.size,
                    'numTilesInGrid':self.numTilesInGrid,
                    'sC':self.sC, 'sV':self.sV, 'sP':self.sP,
                    'numColors':self.numColors, 'numStates':self.numStates, 'boundaryReMap':self.boundaryReMap,
                    'minGen':self.minGen, 'maxGen':self.maxGen, 'fitGen':self.fitGen, 'printGen':self.printGen,
                    'isValued':self.isValued, 'initialValue':self.initialValue, 'valRatio':self.valRatio,
                    'isBoundaried':self.isBoundaried, 'boundaryApprox':self.boundaryApprox,
                    'gol':self.gol, 'overide':self.overide,
                    'tileOutline':self.tileOutline, 'alpha':self.alpha, 'manualCols':self.manualCols,
                    'boundToPC':self.boundToPC, 'boundToCol':self.boundToCol,
                    'colors':self.colors,
                    'borderSet':list(self.borderSet), 'borderVal':self.borderVal, 'borderColor':self.borderColor,
                    'invalidSets':invalidSetsRep, 'invalidVals':self.invalidVals, 'invalidColrs':self.invalidColors
                   }
        with open(self.detailedInfoTrashPath, 'w') as fw:
            json.dump(tileData, fw, indent=4)
            fw.close()
        #readTilingInfo(self.detailedInfoTrashPath)
    def genColAvg(self):
        avgGens = []
        for tilingInd in range(len(self.colValDicts)):
            totalAvgGen = 0
            for color in self.colors:
                color = tuple(color) if type(color) is list else color
                if color in self.colValDicts[tilingInd]:
                    totalAvgGen += self.colValDicts[tilingInd].get(color)
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
            avgGenDiff = totalGenDiff / (len(self.colors)-1)
            avgGenDiffs.append(avgGenDiff)
        return avgGenDiffs

    #######################
    ## Save All And Exit ##
    #######################
    def saveAndExit(self):
        if self.captureStatistics:
            self.saveStabilityFig()
            self.saveValFig()
        if self.isBoundaried:
            self.saveBoundaryFig()
        if not self.gol and self.captureStatistics:
            self.saveColorCompFig()
            self.saveNormColCompFig()
            self.saveAvgGenDiffFig()
            self.saveFitnessFig()
        plt.close('all')
        del self.animatingFigure
        if self.ga:
            ## Relocate fit tilings
            if self.ptIndex > self.fitGen:
                files = glob.glob(self.localTrashPath)
                for f in files:
                    shutil.move(f, self.localPath)
                if not self.silence:
                    print('Files relocated succesfully')
        if not self.silence:
            ## Print execution completion and time
            print('-'*50)
            print('Algorithm Completed')
            print('Executed in {} seconds'.format(time.time()-self.startTime))
            print('-'*50)
            print(' '*50)
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
    with open(path, 'r') as fr:
        tilingInfo = json.load(fr)
        fr.close()
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
                
def genColors(manualCols, numColors):
    colors = []
    if manualCols or numColors>19:
        ## Manually create colors
        # (hue, saturation, value)
        hsvCols = [(x/numColors, 1, 0.75) for x in range(numColors)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvCols))
    else:
        ## Classic colors
        colors = sns.color_palette("bright", numColors)
        #colors = sns.color_palette("husl", numColors)
        #colors = sns.color_palette("cubehelix", numColors)

        ## Divergent colors
        #colors = sns.color_palette("BrBG", numColors)
        #colors = sns.color_palette("coolwarm", numColors)

        ## Gradient colors
        #colors = sns.cubehelix_palette(numColors, dark=0.1, light=0.9)
    return colors
            
def genBounds(boundaryReMap, numStates, numColors, colors):
    sampleDef = 1000
    upper = 1
    bounds, boundToCol, boundToKeyCol, boundToPC = [], {}, {}, {}
    if boundaryReMap:
        upper = 4
    if numStates==1:
        bounds = [numStates]
        boundToCol, boundToKeyCol, boundToPC = {}, {}, {}
        boundToCol[0] = colors[0]
        boundToKeyCol = boundToCol
        boundToPC[0] = rd.randrange(0, sampleDef+1)/sampleDef
    else:
        samplePop = range(1, numStates)
        sample = rd.sample(samplePop, numColors-1)
        sample.append(numStates)
        bounds = sorted(sample)
        boundToCol, boundToKeyCol, boundToPC = {}, {}, {}
        for i, bound in enumerate(bounds):
            boundToCol[bound] = colors[i]
            #self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
            if rd.randrange(0, upper) == 3:
                boundToKeyCol[bound] = colors[rd.randrange(0, len(colors))]
                boundToPC[bound] = 0
            else:
                boundToKeyCol[bound] = colors[i]
                boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
    return bounds, boundToCol, boundToPC
                
                
                


###############################
## Local Animation Execution ##
###############################
## Main definition
def main():
    #p = cProfile.Profile()
    # Enable profiling
    #p.enable()
    
    cleanFileSpace(True, fitClean=False, unfitClean=True)
    ## Space properties ##
    dim = 5
    size = 5
    #Shift vector properties
    sC = 0
    sV = [0, 0.1, 0.2, 0.3, 0.4]
    shiftZeroes, shiftRandom, shiftByHalves = True, False, False
    sP = (shiftZeroes, shiftRandom, shiftByHalves)
    
    ## Physics ##
    numColors = 20
    numStates = 10000
    boundaryReMap = True
    #Display options
    manualCols = True
    tileOutline = True
    alpha = 1

    ## valuedRandomly, valuedByDim, valuedBySize, valuedByTT
    initialValue = (True, False, False, False)

    minGen = 20
    maxGen = 30
    fitGen = 31
    printGen = 0

    isBoundaried = True
    ## Setting boundary approx trades time complexity for calculating the exact tiling
    ## Setting boundaryApprox as True improves time complexity and gives tiling approximation
    boundaryApprox = False

    ## Change gamemode to GOL
    gol = False

    ## Overide ensures maxGen generations
    overide = False

    borderSet = {0, 1, 2, 3, 4, 5, 6}
    
    invalidSets = []

    invalidSet = set()
    invalidSet2 = set()
    # for a in range(-size, size+1):
    #     for b in range(-size, size+1):
    #         invalidSet.add((0, a, 1, b))
    # for r in range(dim):
    #     for s in range(r+1, dim):
    #         invalidSet.add((r, 0, s, 0))
    #         invalidSet2.add((r, 2, s, 2))
            
    invalidSet.update([
        # (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 2), (0, 2, 1, 0)
        ])
    # invalidSets.append(invalidSet)
    # invalidSets.append(invalidSet2)
    borderColor = 'black'
    invalidColors = []
    #invalidColors = ['black', 'black']
    borderVal = 0
    invalidVals = []
    invalidVals = [numStates, numStates]
    dispBorder = True
    dispInvalid = True
    
    captureStatistics = True
    
    
    maxFit = float('-inf')
    bestTilingName = ''
    numIterations = 1
    for iterationNum in range(numIterations):
        currMultigrid = MultigridList(dim, size, sC=sC, sV=sV, sP=sP,
                                minGen=minGen, maxGen=maxGen, fitGen=fitGen, printGen=printGen,
                                numColors=numColors, manualCols=manualCols, numStates=numStates,
                                isValued=True, initialValue=initialValue, valRatio=0.5,
                                isBoundaried=isBoundaried, boundaryReMap=boundaryReMap, boundaryApprox=boundaryApprox,
                                gol=gol, tileOutline=tileOutline, alpha=alpha, overide=overide,
                                borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
                                borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid,
                                iterationNum=iterationNum, captureStatistics=captureStatistics)
        if not gol and captureStatistics and currMultigrid.tilingFitness > maxFit:
            bestTilingName = currMultigrid.multigridListInd
            maxFit = currMultigrid.tilingFitness
    
    bestTilingName = 'No Statistics Captured' if not bestTilingName else bestTilingName
    print(bestTilingName)

    #p.disable()
    #p.print_stats()
    # Dump the stats to a file
    #p.dump_stats("results.prof")

## Local main call
if __name__ == '__main__':
    main()



## TODO: If dim is eve, each iteration over multigrid has two not one invalid tilings, thats why we get some lines and not rhombs i think
## TODO: Fix sV
## TODO: Fix tile type