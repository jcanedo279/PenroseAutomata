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
class DijkstraMultigrid:
    ##################
    ## Init Methods ##
    ##################
    def __init__(self, dim, size,
                 sC=0, sV=None, sP=(False,True,True),
                 isValued=True,
                 maxGen=10, printGen=0,
                 tileOutline=False, alpha=1,
                 borderSet={0,1,2,3,4,5,6}, borderColor='black', borderVal=-1, dispBorder=False,
                 invalidSets=[], invalidColors=[], invalidVals=[], dispInvalid=True,
                 iterationNum=0,
                 ga=False, gaPath=None):
        
        self.source = (0, 1, 1, 0)
        
        self.target = (1, 0, 2, 1)
        
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
        
        self.silence = True if ga else False
        
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
        self.currentGrid.invalidSet = self.invalidSet

        self.animatingFigure = plt.figure()
        self.ax = plt.axes()
        self.ax.axis('equal')

        self.anim = FuncAnimation(self.animatingFigure, self.updateAnimation, frames=self.genFrames, init_func=self.initPlot(), repeat=False, save_count=self.maxGen)
        self.updateDir()
        if not self.silence:
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
    ########################################
    def updateAnimation(self, i):
        if self.iterInd==0:
            self.currentGrid.genTilingVerts()
            if not self.silence:
                print('Tile vertices generated')
            self.currentGrid.genTileNeighbourhoods()
            self.currentGrid.genNonDiagTileNeighbourhoods()
            if not self.silence:
                print('Tile neighbourhood generated')
            self.currentGrid.ax.cla()
            ## Init Dij Here
            self.currentGrid.minDistances = {tuple(tileInd):float('inf') for tileInd in self.currentGrid.allTiles}
            self.currentGrid.minDistPrev = {tuple(tileInd):None for tileInd in self.currentGrid.allTiles}
            self.currentGrid.minDistances[self.source] = 0
            self.currentGrid.source = self.source
            self.currentGrid.target = self.target
            ## Maybe add marker to source here by changing self.minDistPrev[self.source] to string saying 'source'
                
            self.currentGrid.unVisited = set()
            for tInd in self.currentGrid.allTiles:
                t = self.currentGrid.multiGrid[tInd[0]][tInd[1]][tInd[2]][tInd[3]]
                if len(t.neighbourhood) in self.borderSet or tuple(tInd) in self.invalidSet:
                    continue
                self.currentGrid.unVisited.add(tuple(tInd))
            self.currentGrid.currTInd = self.source
            self.iterInd += 1

        else:
            
            ## Implement Dij Here
            if self.currentGrid.unVisited:
                ## Get Next, this pops next node
                currTile = self.currentGrid.multiGrid[self.currentGrid.currTInd[0]][self.currentGrid.currTInd[1]][self.currentGrid.currTInd[2]][self.currentGrid.currTInd[3]]
                self.currentGrid.unVisited.remove(self.currentGrid.currTInd)
                for nInd in currTile.neighbourhood:
                    n = self.currentGrid.multiGrid[nInd[0]][nInd[1]][nInd[2]][nInd[3]]
                    if len(n.neighbourhood) in self.borderSet or tuple(nInd) in self.invalidSet:
                        continue
                    currDist = self.currentGrid.minDistances[self.currentGrid.currTInd] + 1
                    #print(self.minDistances[tuple(nInd)])
                    if currDist < self.currentGrid.minDistances[tuple(nInd)]:
                        self.currentGrid.minDistances[tuple(nInd)] = currDist
                        self.currentGrid.minDistPrev[tuple(nInd)] = self.currentGrid.currTInd
                        
            
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
            
            # hsvCols = [(x/len(knownVals), 1, 0.75) for x in range(len(knownVals))]
            # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvCols))
            colors = [tuple(color) for color in sns.cubehelix_palette(len(knownVals))]
            
            
            sortedKnowns = sorted(list(knownVals))
            # colorSets = {colors[i]:listOfValSets[valToListInd.get(val)] for i, val in enumerate(sortedKnowns)}
            colorSets = {}
            for i, val in enumerate(sortedKnowns):
                color = 'white' if val == float('inf') else colors[i]
                colorSets.update({color:listOfValSets[valToListInd.get(val)]})
                
            if self.currentGrid.currTInd == self.target:
                path = set([self.source])
                cur = self.target
                while cur != self.source:
                    path.add(cur)
                    cur = self.currentGrid.minDistPrev[cur]
                colorSets.update({'gold':path})
            
            self.currentGrid.ax.cla()
            nextGrid, axis = self.currentGrid.nextSelective(self.currentGrid.currTInd, colorSets, self.source, self.target)
            
            ## Stop the animation before maxGen
            if self.iterInd > self.maxGen-1:
                self.continueAnimation = False
                
            ## Get next currTInd
            currMin = None
            minVal = float('inf')
            for key, val in self.currentGrid.minDistances.items():
                if val < minVal and val != float('inf') and key in self.currentGrid.unVisited:
                    currMin, minVal = key, val
            nextGrid.currTInd = currMin
            nextGrid.unVisited = self.currentGrid.unVisited
            nextGrid.minDistances, nextGrid.minDistPrev = self.currentGrid.minDistances, self.currentGrid.minDistPrev
            
              
            self.currentGrid = nextGrid
                    
            
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
            if not self.silence:
                print(f'Grid {self.iterInd} complete', end='\r')
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
        self.rootPath = filePath.replace('src/DijkstraMultigrid.py', 'outputData/dijMultigridData/')
        
        self.gridPath = 'fitDisp/'
        self.dijInd = str(rd.randrange(0, 2000000000))
        self.localPath = f"{self.rootPath}dijInd{self.dijInd}/"
            
        ## gifPaths
        self.gifPath = f'{self.localPath}dijAnimation.gif'
        

    # def genPngPaths(self, pngName):
    #     pngPath = f'{self.localPath}listInd{self.dijInd[0:3]}{pngName}.png'
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

def cleanFileSpace(dijClean=False):
    filePath = os.path.realpath(__file__)
    rootPath = filePath.replace('src/DijkstraMultigrid.py', '/outputData/')
    if dijClean:
        files = glob.glob(f'{rootPath}dijMultigridData/*')
        for f in files:
            shutil.rmtree(f)
        time.sleep(1)
            
                
                
                


###############################
## Local Animation Execution ##
###############################
## Main definition
def main():
    cleanFileSpace(dijClean=True)
    
    ## Space properties ##
    dim = 5
    size = 5
    #Shift vector properties
    sC = 0
    sV = [0, 0.1, 0.2, 0.3, 0.4]
    shiftZeroes, shiftRandom, shiftByHalves = True, False, False
    sP = (shiftZeroes, shiftRandom, shiftByHalves)
    
    isValued = False
   
    #Display options
    tileOutline = True
    alpha = 1

    maxGen = 200
    printGen = 0


    borderSet = {0, 1, 2, 3, 4, 5, 6}
    borderColor = 'black'
    borderVal = 0
    dispBorder = True
    


    invalidSet = set()
    #invalidSet2 = set()
    for r in range(dim):
        for s in range(r+1, dim):
            invalidSet.add((r, size, s, size))
            invalidSet.add((r, size, s, -size))
            invalidSet.add((r, -size, s, size))
            invalidSet.add((r, -size, s, -size))
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
    invalidColors = ['grey']
    #invalidColors = ['black', 'black']
    invalidVals = []
    #invalidVals = [numStates, numStates]
    dispInvalid = True
    
    
    
    currMultigrid = DijkstraMultigrid(dim, size, sC=sC, sV=sV, sP=sP,
                                    maxGen=maxGen, printGen=printGen,
                                    isValued=isValued,
                                    tileOutline=tileOutline, alpha=alpha,
                                    borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
                                    borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid,
                                    iterationNum=0)

## Local main call
if __name__ == '__main__':
    main()