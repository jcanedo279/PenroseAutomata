import time
import os
import sys
import glob
import shutil

import json

import math

import itertools

import colorsys

import random as rd

import numpy as np

import seaborn as sns

import plotly.graph_objects as go

from pathlib import Path

from MultigridList import MultigridList


class GeneticAutomata:
    def __init__(self, numColors, numStates, manualCols, boundaryReMap,
                 popSize, poolSize,
                 numGens):
        # space = [dim, size]
        # spaceCond = [sC, sV, sP]
        # eng = [numColors, numStates, boundaryReMap]
        # ## initV
        # phys = [b, bTC, bTPC]
        # generations = [30, 40, 41, 0] #0 for now, eventually 41
        # boundary = [True, False]
        # misc = [False, False]
        # border = [borderSet, borderColor, borderVal, dispBorder]
        # invalid = [[], [], [], False]
        # display = [True, 1]
        # ## cS # captureStatistics
        # gene = [space, spaceCond, eng, initialValue, phys, generations, boundary, misc, display, border, invalid, True]
        
        filePath = os.path.realpath(__file__)
        self.poolPath = filePath.replace('src/GeneticAutomata.py', 'outputData/poolMultigridData/')
        
        self.gaInd = str(rd.randrange(0, 2000000000))
        self.gaPath = f'{self.poolPath}gaInd{self.gaInd}/'

        self.gaStatsPath = f'{self.poolPath}gaStats/'
        if not os.path.isdir(self.gaStatsPath):
            os.makedirs(self.gaStatsPath)
        self.gaFitnessesPath = f'{self.gaStatsPath}gaInd{self.gaInd}Fitnesses.png'
        
        self.fittestPath = f'{self.poolPath}fittest.json'
    
        self.numColors, self.numStates = numColors, numStates
        self.manualCols = manualCols
        self.boundaryReMap = boundaryReMap
        self.popSize, self.poolSize = popSize, poolSize
        self.numGens = numGens
        
        self.sampleDef = 1000
        
        self.genColors()
        self.genBounds(evenlyDist=True)
        
        self.popFitnesses = []
        
        self.genInd = 0
        
        if Path(self.fittestPath).is_file():
            self.genPopFromSave()
            print(f"Init pop and pool generated from save     avgPopFit{self.genInd+1}={round(self.avgPopFit, 6)}")
        else:
            self.genPop()
            print('Init pop generated')
            self.evaluatePop()
            print(f"Init pop evaluated     avgPopFit{self.genInd+1}={round(self.avgPopFit, 6)}")
            self.genPool()
            print('Init pool generated')
            print('-'*50)
            self.genInd += 1
    
        self.popIter()
        
        self.saveGAFitnessesFig()
        
            
        
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
            
    def genBounds(self, evenlyDist=False):
        self.bounds, self.boundToCol, self.boundToKeyCol = [], {}, {}
        if self.numStates==1:
            self.bounds = [self.numStates]
            self.boundToCol, self.boundToKeyCol = {}, {}
            self.boundToCol[0] = self.colors[0]
            self.boundToKeyCol = self.boundToCol
        else:
            upper = 4 if self.boundaryReMap else 1
            if evenlyDist:
                if self.numStates%self.numColors == 0:
                    step = self.numStates/(self.numColors-1)
                    for i in range(self.numColors-1):
                        self.bounds.append(int(i*step))
                    self.bounds.append(self.numStates)
                    for i, bound in enumerate(self.bounds):
                        self.boundToCol[bound] = self.colors[i]
                        #self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
                        if rd.randrange(0, upper) == 3:
                            self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]
                        else:
                            self.boundToKeyCol[bound] = self.colors[i]
                else:
                    sys.exit('numStates must be dividible by numCols')
            else:
                samplePop = range(1, self.numStates)
                sample = rd.sample(samplePop, self.numColors-1)
                sample.append(self.numStates)
                self.bounds = sorted(sample)
                for i, bound in enumerate(self.bounds):
                    self.boundToCol[bound] = self.colors[i]
                    #self.boundToPC[bound] = rd.randrange(0, sampleDef+1)/sampleDef
                    if rd.randrange(0, upper) == 3:
                        self.boundToKeyCol[bound] = self.colors[rd.randrange(0, len(self.colors))]
                    else:
                        self.boundToKeyCol[bound] = self.colors[i]
                    
    def genPop(self):
        self.curPop = []
        for _ in range(self.popSize):
            geneFrame = self.genGeneFrame()
            randGene = self.genGene(geneFrame)
            self.curPop.append(randGene)
    
    def evaluateAutomata(self, gene, displayTiling=False):
        printTiling = 0 if displayTiling else gene[5][3]
        dispBorder = True if displayTiling else gene[9][3]
        gaPath = f'{self.gaPath}gaGen{self.genInd}/'
        geneMultigrid = MultigridList(gene[0][0], gene[0][1],
                                      sC=gene[1][0], sV=gene[1][1], sP=gene[1][2],
                                      isValued=True, valRatio=0.5,
                                      numColors=gene[2][0], numStates=gene[2][1], boundaryReMap=gene[2][2],
                                      colors=self.colors, initialValue=gene[3],
                                      bounds=gene[4][0], boundToCol=gene[4][1], boundToPC=gene[4][2],
                                      minGen=gene[5][0], maxGen=gene[5][1], fitGen=gene[5][2], printGen=printTiling,
                                      isBoundaried=gene[6][0], boundaryApprox=gene[6][1],
                                      gol=gene[7][0], overide=gene[7][1],
                                      tileOutline=gene[8][0], alpha=gene[8][1],
                                      borderSet=gene[9][0], borderColor=gene[9][1], borderVal=gene[9][2], dispBorder=dispBorder,
                                      invalidSets=gene[10][0], invalidColors=gene[10][1], invalidVals=gene[10][2], dispInvalid=gene[10][3],
                                      iterationNum=0, captureStatistics=gene[11], ga=True,
                                      gaPath=gaPath)
        return geneMultigrid.avgFit
          
    def genGeneFrame(self):
        space = [5, 5]
        spaceCond = [None, None, None]
        eng = [self.numColors, self.numStates, self.boundaryReMap]
        initV = (True, False, False, False)
        phys = [self.bounds, self.boundToCol, {}]
        generations = [30, 40, 41, 41]
        boundary = [True, False]
        misc = [False, False]
        border = [{0, 1, 2, 3, 4, 5, 6}, 'black', 0, False]
        invalid = [[], [], [], False]
        display = [True, 1]

        geneFrame = [space, spaceCond, eng, initV, phys, generations, boundary, misc, display, border, invalid, True]
        return geneFrame
            
    def genGene(self, geneFrame):
        gene = geneFrame.copy()
        # gene[0][0], gene[0][1] = rd.randrange(5, 11), rd.randrange(5, 11)
        sPs = [(True, False, False), (False, True, False), (False, False, True)]
        gene[1][0], gene[1][1], gene[1][2] = rd.randrange(-self.sampleDef, self.sampleDef)/self.sampleDef, None, sPs[rd.randrange(0, 3)]
        iVs = [(False, False, False, True), (False, False, True, False), (False, True, False, False), (True, False, False, False)]
        gene[3] = iVs[rd.randrange(0, 4)]
        for bTPCKey in gene[4][1].keys():
            gene[4][2][bTPCKey] = rd.randrange(0, self.sampleDef)/self.sampleDef
        # gene[9][2] = rd.randrange(-gene[2][1]*self.sampleDef, gene[2][1]*self.sampleDef)/self.sampleDef
        return gene   
    
    def evaluatePop(self):
        self.curFitnesses = {}
        self.popFit = 0
        for i, gene in enumerate(self.curPop):
            geneFit = self.evaluateAutomata(gene)
            self.popFit += geneFit
            self.curFitnesses[geneFit] = gene
            print(f'Tiling {i+1}/{self.popSize} evaluated          avgTilingFit{i+1}={geneFit}', end='\r')
        self.avgPopFit = self.popFit/self.popSize
        self.popFitnesses.append(self.avgPopFit)
        print(' '*75, end='\r')
        print('Displaying Fittest...', end='\r')
        self.evaluateAutomata(self.curFitnesses[max(self.curFitnesses.keys())], displayTiling=True)
        print(' '*50, end='\r')
        print(f'Fittest tiling in generation {self.genInd+1} displayed')
        self.dumpFittest(max(self.curFitnesses.keys()), self.curFitnesses[max(self.curFitnesses.keys())])
        
    def genPool(self):
        fitnesses = [fit**2 for fit in self.curFitnesses.keys()]
        totalFitness = sum(fitnesses)
        fitBounds = []
        fitBounds.append(fitnesses[0])
        for i in range(1, len(fitnesses)):
            fitBounds.append(fitBounds[i-1]+fitnesses[i])
        self.pool = []
        for _ in range(self.poolSize):
            rdVal = rd.randrange(0, int(totalFitness*self.sampleDef)+1)/self.sampleDef
            gene = self.curFitnesses.get(math.sqrt(fitnesses[np.searchsorted(fitBounds, rdVal)]))
            self.pool.append(gene)
        
    def popIter(self):
        while(self.genInd < self.numGens):
            self.currPop = self.crossPool()
            print(f'Gen {self.genInd+1}/{self.numGens} crossed')
            self.curPop = [self.mutateGene(gene) for gene in self.curPop]
            print(f'Gen {self.genInd+1}/{self.numGens} mutated')
            self.evaluatePop()
            print(f"Gen {self.genInd+1}/{self.numGens} evaluated     avgPopFit{self.genInd+1}={round(self.avgPopFit, 6)}")
            self.genPool()
            print(f'Gen {self.genInd+1}/{self.numGens} pooled')
            print('-'*50)
            self.genInd += 1
            
    def crossPool(self):
        poolIter = iter(self.pool)
        nextPop = []
        popCnt = 0
        for gene1 in poolIter:
            if popCnt >= self.popSize:
                break
            gene2 = next(poolIter)
            popCnt += 2
            nextG1, nextG2 = self.crossGenes(gene1, gene2)
            nextPop.append(nextG1)
            nextPop.append(nextG2)
        return nextPop
                 
    def crossGenes(self, gene1, gene2):
        g1, g2 = gene1.copy(), gene2.copy()
        ## Cross dim and Size
        # if rd.randrange(0, self.sampleDef)/self.sampleDef < 0.125:
        #     g1[0][0], g1[0][1] = gene2[0][0], gene2[0][1]
        #     g2[0][0], g2[0][1] = gene1[0][0], gene1[0][1]
        ## Cross sC and sP
        if rd.randrange(0, self.sampleDef)/self.sampleDef < 0.125:
            g1[1][0], gene1[1][2] = gene2[1][0], gene2[1][2]
            g2[1][0], gene2[1][2] = gene1[1][0], gene1[1][2]
        ## Cross initVal
        if rd.randrange(0, self.sampleDef)/self.sampleDef < 0.125:
            g1[3] = gene2[3]
            g2[3] = gene1[3]
        ## Cross bTPC
        if rd.randrange(0, self.sampleDef)/self.sampleDef < 0.125:
            for bTPCKey in gene1[4][2].keys():
                g1[4][2][bTPCKey] = gene2[4][2][bTPCKey]
                g2[4][2][bTPCKey] = gene1[4][2][bTPCKey]
        ## Cross boundaryValue
        # if rd.randrange(0, self.sampleDef)/self.sampleDef < 0.75:
        #     g1[9][2] = gene2[9][2]
        #     g2[9][2] = gene1[9][2]
        return g1, g2
  
    def mutateGene(self, gene):
        ## Update dimmension and size
        # gene[0][0], gene[0][1] = int(self.mutateVal(gene[0][0], 5, 10, 1, 0.125, 4)), int(self.mutateVal(gene[0][1], 5, 10, 1, 0.125, 4))
        ## Update sC and sP
        gene[1][0], gene[1][2] = self.mutateVal(gene[1][0], -1, 1, 0.125, 0.125, 6), self.mutateFromSet(gene[1][2], {(True, False, False),
                                                                                                                   (False, True, False)}, 0.125)
        gene[1][1] = None # Delete the old shiftVect
        ## Update boundaryReMap
        #gene[2][2] = self.mutateFromSet(gene[2][2], {False, True}, 0.5)
        ## Update initialValue
        gene[3] = self.mutateFromSet(gene[3], {(False, False, True, False), (False, True, False, False), (True, False, False, False)}, 0.125)
        ## Update bTPC
        for bTPCKey, bTPCVal in gene[4][2].items():
            gene[4][2][bTPCKey] = self.mutateVal(bTPCVal, 0, 1, 0.0001, 0.125, 7, btc=True)
        ## Update boundaryValue
        # gene[9][2] = self.mutateVal(gene[9][2], -gene[2][1], gene[2][1], 1, 0.125/2, 5)
        return gene
        
    def mutateVal(self, val, start, stop, sampDef, mutationChance, mutationForm, btc=False):
        newVal = val
        if mutationChance == 0:
            mutationChance = 0.0001
        ## We generate a range of numbers of defenition 0.0001, if mutationRate is less than one val is mutated,
        ## Furthermore if mutationRate is less than 
        mutationPick = rd.randrange(0, 1+int(((1/mutationChance)-1)*self.sampleDef), 1)/self.sampleDef
        mutationAmount = ((rd.randrange(0, self.sampleDef, 1))/self.sampleDef)**mutationForm ## mutForm makes nearby values more likely
        if mutationPick <= (val-start)/(stop-start):
            ## Mutate down
            newVal = val - ((val-start)*mutationAmount)
        elif mutationPick <= 1:
            ## Mutate up
            newVal = val + ((stop-val)*mutationAmount)
        if not btc:
            newVal = round(newVal*float(sampDef))/sampDef
            newVal = max(start, newVal)
            newVal = min(newVal, stop)
        return newVal
    
    def mutateFromSet(self, val, valSet, mutationChance):
        currSet = valSet.copy()
        setL = list(currSet.symmetric_difference({val}))
        if rd.randrange(0, 1000, 1)/1000 <= mutationChance:
            return setL[rd.randrange(0, len(setL))]
        return val
    
    def dumpFittest(self, fittestVal, fittestGene):
        fittestGene[9][0] = list(fittestGene[9][0])
        nextFittestDict = {}
        if Path(self.fittestPath).is_file():
            with open(self.fittestPath, 'r') as fr:
                nextFittestDict = json.load(fr)
                nextFittestDict[fittestVal] = fittestGene
                fr.close()
        else:
            nextFittestDict = {fittestVal:fittestGene}
        with open(self.fittestPath, 'w') as fw:
            json.dump(nextFittestDict, fw, indent=4)
            fw.close()
    
    def genPopFromSave(self):
        if Path(self.fittestPath).is_file():
            ## Load data
            saveDict = {}
            with open(self.fittestPath, 'r') as fr:
                saveDict = json.load(fr)
                fr.close()
            saveFitnesses, saveGenes = [], []
            for key, val in saveDict.items():
                saveFitnesses.append(float(key))
                saveGenes.append(val)
            #saveFitnesses, saveGenes = zip(*sorted(zip(saveFitnesses, saveGenes)))
            ## Gen fitness map
            totalFitness = sum(saveFitnesses)
            fitBounds = []
            fitBounds.append(saveFitnesses[0])
            for i in range(1, len(saveFitnesses)):
                fitBounds.append(fitBounds[i-1]+saveFitnesses[i])
            ## 
            self.popFit = 0
            self.pool, self.curPop, self.curFitnesses = [], [], []
            for i in range(self.poolSize):
                rdVal = rd.randrange(0, int(totalFitness*self.sampleDef)+1)/self.sampleDef
                rdFitness = saveFitnesses[np.searchsorted(fitBounds, rdVal)]
                gene = saveDict.get(str(rdFitness))
                
                gene[1][2] = tuple(gene[1][2])
                gene[3] = tuple(gene[3])
                gene[4][0] = [int(bound) for bound in gene[4][0]]   
                gene[4][1] = {int(key):val for key,val in gene[4][1].items()}
                gene[4][2] = {int(key):val for key,val in gene[4][2].items()}
        
                if i < self.popSize:
                    self.popFit += rdFitness
                    self.popFitnesses.append(rdFitness)
                    self.curFitnesses.append(rdFitness)
                    self.curPop.append(gene)
                self.avgPopFit = self.popFit/self.popSize
                self.pool.append(gene)
                
            self.genInd += 1

        else:
            print('No save found')
    
    def saveGAFitnessesFig(self):
        gaFitFig = go.Figure()
        x = [str(i) for i in range(len(self.popFitnesses))]
        gaFitFig.add_trace(go.Scatter(x=x, y=self.popFitnesses, name='pop fitness', line=dict(color='firebrick', width=4)))
        gaFitFig.update_layout(title=f'Pop Fitness vs. Tile Generation',
                                  xaxis_title='tile Index',
                                  yaxis_title='fitness of population')
        gaFitFig.write_image(self.gaFitnessesPath)
        

            




def readTilingInfo(path):
    with open(path, 'r') as fr:
        tilingInfo = json.load(fr)
    return tilingInfo

def cleanFileSpace(cleaning, fitClean=False, unfitClean=False, poolClean=False):
    filePath = os.path.realpath(__file__)
    rootPath = filePath.replace('src/GeneticAutomata.py', 'outputData/')
    if cleaning:
        if fitClean:
            files = glob.glob(f'{rootPath}fitMultigridData/*')
            for f in files:
                shutil.rmtree(f)
        if unfitClean:
            tFiles = glob.glob(f'{rootPath}unfitMultigridData/*')
            for tF in tFiles:
                shutil.rmtree(tF)
        if poolClean:
            tFiles = glob.glob(f'{rootPath}poolMultigridData/*')
            for tF in tFiles:
                os.remove(tF)
                
                
                
                


###############################
## Local Animation Execution ##
###############################
## Main definition
def main():
    #p = cProfile.Profile()
    # Enable profiling
    #p.enable()
    
    cleanFileSpace(True, fitClean=False, unfitClean=True, poolClean=False)

    ## Physics ##
    numColors = 20
    numStates = 10000
    
    manualCols = True
    boundaryReMap = False
    
    popSize = 3
    poolSize = 5
    
    numGens = 2
    
    GeneticAutomata(numColors, numStates, manualCols, boundaryReMap, popSize, poolSize, numGens)
    
    
    # maxFit = float('-inf')
    # bestTilingName = ''
    # numIterations = 1
    # for iterationNum in range(numIterations):
    #     currMultigrid = MultigridList(dim, size, sV, sC=sC, sP=sP,
    #                             minGen=minGen, maxGen=maxGen, fitGen=fitGen, printGen=printGen,
    #                             numColors=numColors, manualCols=manualCols, numStates=numStates,
    #                             isValued=True, initialValue=initialValue, valRatio=0.5,
    #                             isBoundaried=isBoundaried, boundaryReMap=boundaryReMap, boundaryApprox=boundaryApprox,
    #                             gol=gol, tileOutline=tileOutline, alpha=alpha, overide=overide,
    #                             borderSet=borderSet, invalidSets=invalidSets, borderColor=borderColor, invalidColors=invalidColors,
    #                             borderVal=borderVal, invalidVals=invalidVals, dispBorder=dispBorder, dispInvalid=dispInvalid,
    #                             iterationNum=iterationNum, captureStatistics=captureStatistics)
    #     if not gol and captureStatistics and currMultigrid.tilingFitness > maxFit:
    #         bestTilingName = currMultigrid.multigridListInd
    #         maxFit = currMultigrid.tilingFitness
    
    # bestTilingName = 'No Statistics Captured' if not bestTilingName else bestTilingName
    # print(bestTilingName)

    #p.disable()
    #p.print_stats()
    # Dump the stats to a file
    #p.dump_stats("results.prof")

## Local main call
if __name__ == '__main__':
    main()
    
    
    
    
    
## TODO: genFromSaveFile param safety check to check that all genes are proper (ie have the same number of states as the current script)
## If they are different, we delete and start from scratch, if they are the same, we gen from save