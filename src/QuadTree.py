import shutil
import glob
import math
import time

import numpy as np
import random as rd

from pprint import pprint

from MultigridList import MultigridList


class QuadNode:
    def __init__(self, val, sideLen, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.sideLen = sideLen
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
        self.isCompletelyHashed = False

class QuadTree:
    def __init__(self, size,
                 sC,
                 numStates, numColors,
                 initVal,
                 minGen, maxGen, fitGen, printGen,
                 tileOutline,
                 borderSet={0,1,2,3,4,5,6}, borderColor='black', borderVal=0, dispBorder=True):
        self.dim, self.size = 4, size
        
        self.sC = sC
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = True, False, False
        shiftProperties = (self.shiftZeroes, self.shiftRandom, self.shiftByHalves)

        self.numStates, self.numColors = numStates, numColors
        
        self.initVal=initVal
        
        self.minGen, self.maxGen, self.fitGen, self.printGen = minGen, maxGen, fitGen, printGen

        self.tileOutline = tileOutline
        
        self.borderSet, self.borderColor, self.borderVal, self.dispBorder = borderSet, borderColor, borderVal, dispBorder

        self.tileSize = 10

        ## These paths are required for the MultigridList object to instantiate
        self.rootPath = 'MultigridListData/'
        self.multigridListInd = rd.randrange(0, 2000000000)
        self.gridPath = 'listInd' + str(self.multigridListInd) + '/'
        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('MultigridListData', 'TrashLists')


        self.knownNodes = set()
        self.quadToNode = dict() 
        
        self.mList = MultigridList(self.dim, self.size,
                                   sC=self.sC, sP=shiftProperties,
                                   numColors=self.numColors, numStates=self.numStates,
                                   initialValue=self.initVal,
                                   minGen=self.minGen, maxGen=self.maxGen, fitGen=self.fitGen, printGen=self.printGen,
                                   tileOutline=self.tileOutline,
                                   borderSet=self.borderSet, borderColor=self.borderColor, borderVal=self.borderVal, dispBorder=self.dispBorder,
                                   iterationNum=0, captureStatistics=True)
        
        
        ## Store the grid locally inside of this object, this avoids messing with the generations in MultigridList
        self.origGrid = self.mList.currentGrid
        ## Uses genValMap, findRootTile, and findNeighbour to convert gridList's projected tiling into a valued 2d list
        self.genValMap()
        ## This is the call to the divide and conquer recursive function subdivideGrid, returns the root of the quadTree
        self.quadRoot = self.makeQuadTree()



    def genValMap(self):
        ## From the following method we find the analytical form of rootTileInd in four dimmensions
        #rootTileInd = findRootTile()
        rootTileInd = [0, -self.size, 1, -self.size]
        currTileInd = rootTileInd
        
        if not self.borderSet=={0,1,2,3,4,5,6}:
            playableSideLen = 0
            while currTileInd!=[1, -self.size, 2, -self.size]:
                playableSideLen += 1
                currTileInd = self.findNeighbour(currTileInd, rightN=True)
        else:
            ## Skip over the border value
            currTileInd = self.findNeighbour(currTileInd, bottomRightN=True)

            playableSideLen = 0
            while not len(self.origGrid.multiGrid[currTileInd[0]][currTileInd[1]][currTileInd[2]][currTileInd[3]].neighbourhood) in self.borderSet:
                playableSideLen += 1
                currTileInd = self.findNeighbour(currTileInd, rightN=True)


        maxDepth = math.floor(math.log(playableSideLen, 2))
        self.maxSideLen = 2**maxDepth

        numTilesTilCentered = int((playableSideLen-self.maxSideLen)/2)

        for _ in range(numTilesTilCentered):
            firstPlayableTileInd = self.findNeighbour(rootTileInd, bottomRightN=True)

        self.valMap = [[0 for x in range(self.maxSideLen)] for row in range(self.maxSideLen)]
        currTileInd = firstPlayableTileInd
        ## Iterate down the grid
        for i in range(self.maxSideLen):
            currTileInd = self.findNeighbour(currTileInd, bottomN=True)
            currRowTileInd = currTileInd
            ## Iterate across the grid
            for j in range(self.maxSideLen):
                currRowTileInd = self.findNeighbour(currRowTileInd, rightN=True)
                self.valMap[i][j] = self.origGrid.multiGrid[currRowTileInd[0]][currRowTileInd[1]][currRowTileInd[2]][currRowTileInd[3]].val


    ## From this we found that the top left-most tile has coordinates: [0, -size, 1, -size]
    def findRootTile(self):
        currRootTileInd = [0, 0, 0, 0]
        xMaxLen = float('inf')
        rndVal = 4

        for r in range(self.dim):
            ## We do not want to check the outermost layer as they are non playable
            for a in range(-self.size, self.size+1):
                for s in range(r+1, self.dim):
                    for b in range(-self.size, self.size+1):
                        t1 = self.origGrid.multiGrid[r][a][s][b]
                        for vertex in t1.vertices:
                            if (round(vertex[0], rndVal) == round(vertex[1], rndVal)) and (round(vertex[0], rndVal) < xMaxLen):
                                xMaxLen = round(vertex[0], rndVal)
                                currRootTileInd = [r, a, s, b]               
        return currRootTileInd

    def findNeighbour(self, tileInd, rightN=False, bottomN=False, bottomRightN=False):
        rndVal = 4

        tile = self.origGrid.multiGrid[tileInd[0]][tileInd[1]][tileInd[2]][tileInd[3]]
        txVertAv = sum([round(vert[0], rndVal) for vert in tile.vertices])/4
        tyVertAv = sum([round(vert[1], rndVal) for vert in tile.vertices])/4
        for nInd in tile.neighbourhood:
            n = self.origGrid.multiGrid[nInd[0]][nInd[1]][nInd[2]][nInd[3]]
            nxVertAv = sum([round(vert[0], rndVal) for vert in n.vertices])/4
            nyVertAv = sum([round(vert[1], rndVal) for vert in n.vertices])/4
            if rightN:
                if (nxVertAv > txVertAv) and (nyVertAv == tyVertAv):
                    return nInd
            elif bottomN:
                if (nxVertAv == txVertAv) and (nyVertAv > tyVertAv):
                    return nInd
            elif bottomRightN:
                if (nxVertAv > txVertAv) and (nyVertAv > tyVertAv):
                    return nInd

    '''
        The following functions (makeQuadNode, subdivideGrid, printQuadTreeNodes) specifically concern the divide and conquer aspect
        of the assignment. 
    '''
    def makeQuadNode(self, currQuadrant):
        ## Intent: makeQuadNode is a helper method and serves to check if a quadrant has been seen and whether it is completely hashed.
        # Additionally, makeQuadNode creates and returns the node representative of currQuadrant
        hashQuad = tuple([self.origGrid.valToBound(gridVal) for gridVal in currQuadrant])
        if hashQuad in self.knownNodes:
            return self.quadToNode.get(hashQuad)
        else:
            self.knownNodes.add(hashQuad)

        valueSet = set([self.origGrid.valToBound(gridVal) for gridVal in currQuadrant])
        if len(valueSet) == 1:
            ## If we do not need to subdivide the space, the following parameterized QuadNode is created, and it is compeltely hashed as it is a leaf
            currNode = QuadNode(self.origGrid.valToBound(currQuadrant[0]), math.sqrt(len(currQuadrant)), True, None, None, None, None)
            currNode.isCompletelyHashed = True
        else:
            ## Saving the non-uniform grid to the QuadNode is important if you 
            # want to compress space or hash state changes
            currNode = QuadNode(hashQuad, math.sqrt(len(currQuadrant)), False, None, None, None, None)
        self.quadToNode.update({hashQuad:currNode})
        return self.quadToNode.get(hashQuad)

    def subdivideGrid(self, xi, xf, yi, yf):
        ## Intent: subdivideGrid is a recursive function that takes in a QuadTree object (used for storing a bitmap)
        # and four boundaries (for x and y with start and stop each)

        ## Prec1: All additional functions and data structures such as the QuadTree and QuadNode classes are implemented correctly
        ## Prec2: The input QuadTree object contains valMap (a 2d valued grid) such that it is a maxSideLen x maxSideLen grid containing
        # integer values chosen from a set of numColor boundaries chosen from [0, numStates] (always including numSatates)
        ## Prec3: The four boundaries are integers in the range [0, maxSideLen] where xi<xf and yi<yf
        
        ## Post1: The object node representative of valMap quadrant bounded by the indeces is returned by the function call
        ## Post2: node contains the values of the subgrid bounded by the input boundary indices
        ## Post3: If all the values of the current valMap quadrant are all within a subset of the integers in
        # {xeR: 0<x<numStates} (subset specified by the input QuadTree object), then node is a leaf, otherwise node is not a leaf
        # Disc: If the input boundaries bound only a single cell then the node is a leaf by Post 3
        ## Post4: If the node is not a leaf then it contains references to four child nodes, one for each of the four cardinal directions
        # the current quadrant of valMap is split in four and each child is called with its respective boundaries
        ## Post5: All returned nodes are canonicalized by memoization, previously seen nodes return a reference to the previously created object
        node = self.makeQuadNode([val for row in self.valMap[xi:xf] for val in row[yi:yf]])
        if (not node.isLeaf) and (not node.isCompletelyHashed):
            xc, yc = (xf-xi)//2, (yf-yi)//2
            node.topLeft = self.subdivideGrid(xi, xi+xc, yi, yi+yc)
            node.topRight = self.subdivideGrid(xi, xi+xc, yi+yc, yf)
            node.bottomLeft = self.subdivideGrid(xi+xc, xf, yi, yi+yc)
            node.bottomRight = self.subdivideGrid(xi+xc, xf, yi+yc, yf)
        node.isCompletelyHashed = True
        return node

    def makeQuadTree(self):
        return self.subdivideGrid(0, self.maxSideLen, 0, self.maxSideLen)

    def printQuadTreeNodes(self, currNode, allNodes=False, dispid=False):
        if allNodes:
            print('val={}'.format(currNode.val))
            print('size={}'.format(currNode.sideLen))
            print(id(currNode))
        elif currNode.isLeaf:
            print('val={}, size={}'.format(currNode.val, currNode.sideLen))
            print(id(currNode))
        if not currNode.isLeaf:
            if currNode.bottomLeft != None:
                self.printQuadTreeNodes(currNode.bottomLeft, allNodes=allNodes, dispid=dispid)
            if currNode.bottomRight != None:
                self.printQuadTreeNodes(currNode.bottomRight, allNodes=allNodes, dispid=dispid)
            if currNode.topLeft != None:
                self.printQuadTreeNodes(currNode.topLeft, allNodes=allNodes, dispid=dispid)
            if currNode.topRight != None:
                self.printQuadTreeNodes(currNode.topRight, allNodes=allNodes, dispid=dispid)
        
                
    def multigridSort(self, xi=0, xf=-1, yi=0, yf=-1):
        ## Axiom0: All functions and data structures such as the QuadTree and QuadNode classes are implemented correctly
        # This means that all states (and there are many) that this algorithm accumulates on is taken axiomatically as correct
        # even if that is not so (as it is).
        
        ## Intent: The intent of this function is to sort a sub matrix (defined by the indices xi,xf,yi,yf) on self.valMap, in such a way that
        # repeated elements are efficiently sorted from left to right, then from top to bottom, as if reading lines from a book where the lines are in
        # strictly non-decreasing order. Finally the sorted sub-matrix is displayed on the console.
        
        ## Prec0: self.valMap contains a perfect square matrix (side lengths of n=2^k) with values in the closed interior [0, numStates]
        ## Prec1: xi, xf, yi, yf are valid integer indices of self.valMap in the two grid directors.
        ## Prec2: xf>xi and yf>yi, additionally xf-xi==yf-yi
        
        ## Post0: self.valMap is unaltered
        ## Post1: outMap contains a square matrix with values sorted as described in the intent
        ## Post2: outMap is neatly displayed on the console
        ## Post3: xf>xi and yf>yi, additionally xf-xi==yf-yi
        
        ## Invar0: self.valMap is unaltered
        ## Invar1: xf>xi and yf>yi, additionally xf-xi==yf-yi
        
        ## State0: The final x and y coordinates of the submatrix are set to the maximum dimmension of the matrix if they are -1
        if xf == -1:
            xf = len(self.valMap)
        if yf == -1:
            yf = len(self.valMap)
        ## State1: The matrix is copied to ensure no information is lost, furthermore we only use the submatrix bounded by the input parameters
        valMap = np.array(self.valMap)[xi:xf, yi:yf]
        ## State2: An empty array is used to store the sorted grid
        outArray = [0] * (len(valMap))**2
        ## State3: For each element of the matrix, the item and its respective number of occurences is added to a dictionary valOccurences representing
        # the number of ccurences of each unique value in the sub matrix
        valOccurences = {}
        for row in valMap:
            for item in row:
                if item in valOccurences:
                    valOccurences[item] = valOccurences[item] + 1
                else:
                    valOccurences[item] = 1
        ## State4: The keys of the dictionary (a list of unique values), are sorted (done natively by quicksort)
        sortedKeys = sorted(valOccurences.keys())
        ## State5: The sorted keys are added to the output arrary outArray in their respective number of occurences
        i = 0
        outArray = [0] * (len(valMap))**2
        for value in sortedKeys:
            ## State5.valueInd: For each value in the sorted keys list, the value is added valOccurences[value] times (where valueInd is the index of
            # value in sortedKeys)
            o = 0
            while(o<valOccurences[value]):
                outArray[i] = value
                i += 1
                o += 1
        ## State6: outArray is reshaped using numpy from a 1D list back into a 2D rasterized matrix
        outMat = np.array(outArray).reshape(len(valMap), len(valMap))
        ## State7: The output matrix outMat is displayed on the console, preferably nicely formatted too
        pprint(outMat)
        


def main():
    size = 5
    
    sC = 0

    ## These must both be greater than 4
    numStates = 1000
    numColors = 10
    
    ## Do not change these
    initVal = (True, False, False, False)
    
    ## How many generations into the animation we start the quadTree algorithm
    minGen, maxGen, fitGen, printGen = 0, 0, 11, 0

    tileOutline = True
    
    borderSet, borderColor, borderVal, dispBorder = {0,1,2,3,4,5,6}, 'black', -1, True


    quadTree = QuadTree(size, sC,
                        numStates, numColors,
                        initVal,
                        minGen, maxGen, fitGen, printGen,
                        tileOutline,
                        borderSet=borderSet, borderColor=borderColor, borderVal=borderVal, dispBorder=dispBorder)
    
    #quadTree.printQuadTreeNodes(quadTree.quadRoot, dispid=True)
    
    quadTree.multigridSort(xi=0, xf=5, yi=0, yf=5)
    

if __name__=='__main__':
    main()