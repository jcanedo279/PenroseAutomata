import shutil
import glob
import math
import time

import random as rd

from MultigridTree import MultigridTree


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
    def __init__(self, dim, sC, size, numStates, numColors, tileOutline, alpha, isRadByDim, isRadBySize, maxGen):
        self.dim = dim
        self.sC = sC
        self.size = size

        self.numStates = numStates
        self.numColors = numColors

        self.tileOutline = tileOutline
        self.alpha = alpha
        self.isRadByDim, self.isRadBySize = isRadByDim, isRadBySize
        self.shiftZeroes, self.shiftRandom, self.shiftByHalves = True, False, False
        self.tileSize = 10

        ## These paths are required for the MultigridTree object to instantiate
        self.rootPath = 'MultigridTreeData/'
        self.multigridTreeInd = rd.randrange(0, 2000000000)
        self.gridPath = 'treeInd' + str(self.multigridTreeInd) + '/'
        self.localPath = self.rootPath + self.gridPath
        self.localTrashPath = self.localPath.replace('MultigridTreeData', 'TrashTrees')

        self.maxGen = maxGen

        self.knownNodes = set()
        self.quadToNode = dict()

        self.tree = MultigridTree(self.dim, self.sC, self.size, self.tileSize, self.shiftZeroes, self.shiftRandom, self.shiftByHalves,
                                  self.tileOutline, self.alpha, self.numColors, self.maxGen, isValued=True,
                                  valIsRadByDim=self.isRadByDim, valIsRadBySize=self.isRadBySize, numStates=self.numStates)
        ## Store the grid locally inside of this object, this avoids messing with the generations in MultigridTree
        self.origGrid = self.tree.currentGrid
        ## Uses genValMap, findRootTile, and findNeighbour to convert tree's projected tiling into a valued 2d list
        self.genValMap()
        ## This is the call to the divide and conquer recursive function subdivideGrid, returns the root of the quadTree
        self.quadRoot = self.subdivideGrid(0, self.maxSideLen, 0, self.maxSideLen)



    def genValMap(self):
        #rootTileInd = findRootTile()
        rootTileInd = [0, -self.size, 1, -self.size]
        currTileInd = rootTileInd
        while self.origGrid.multiGrid[currTileInd[0]][currTileInd[1]][currTileInd[2]][currTileInd[3]].val == -1:
            currTileInd = self.findNeighbour(currTileInd, bottomRightN=True)
        firstPlayableTileInd = currTileInd

        playableSideLen = 0
        while self.origGrid.multiGrid[currTileInd[0]][currTileInd[1]][currTileInd[2]][currTileInd[3]].val != -1:
            playableSideLen += 1
            currTileInd = self.findNeighbour(currTileInd, rightN=True)

        maxDepth = math.floor(math.log(playableSideLen, 2))
        self.maxSideLen = 2**maxDepth

        numTilesTilCentered = int((playableSideLen-self.maxSideLen)/2)

        for _ in range(numTilesTilCentered):
            firstPlayableTileInd = self.findNeighbour(firstPlayableTileInd, bottomRightN=True)

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
        The following functions (makeNode, subdivideGrid, printQuadTreeNodes) specifically concern the divide and conquer aspect
        of the assignment. 
    '''
    def makeNode(self, currQuadrant):
        ## Intent: makeNode is a helper method and serves to check if a quadrant has been seen and whether it is completely hashed.
        # Additionally, makeNode creates and returns the node representative of currQuadrant
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
            currNode = QuadNode(currQuadrant, math.sqrt(len(currQuadrant)), False, None, None, None, None)
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
        node = self.makeNode([val for row in self.valMap[xi:xf] for val in row[yi:yf]])
        if (not node.isLeaf) and (not node.isCompletelyHashed):
            xc, yc = (xf-xi)//2, (yf-yi)//2
            node.topLeft = self.subdivideGrid(xi, xi+xc, yi, yi+yc)
            node.topRight = self.subdivideGrid(xi, xi+xc, yi+yc, yf)
            node.bottomLeft = self.subdivideGrid(xi+xc, xf, yi, yi+yc)
            node.bottomRight = self.subdivideGrid(xi+xc, xf, yi+yc, yf)
        node.isCompletelyHashed = True
        return node

    def printQuadTreeNodes(self, currNode, allNodes=False, dispid=False):
        if dispid:
            print(id(currNode))
        if allNodes:
            print('val={}'.format(currNode.val))
            print('size={}'.format(currNode.sideLen))
        elif currNode.isLeaf:
            print('val={}, size={}'.format(currNode.val, currNode.sideLen))
        else:
            if currNode.bottomLeft != None:
                self.printQuadTreeNodes(currNode.bottomLeft)
            if currNode.bottomRight != None:
                self.printQuadTreeNodes(currNode.bottomRight)
            if currNode.topLeft != None:
                self.printQuadTreeNodes(currNode.topLeft)
            if currNode.topRight != None:
                self.printQuadTreeNodes(currNode.topRight)


def main():
    dim = 4
    sC = 0
    size = 30

    numStates = 100
    numColors = 100

    tileOutline = True
    alpha = 1
    isRadByDim, isRadBySize = False, False

    maxGen = 1

    quadTree = QuadTree(dim, sC, size, numStates, numColors, tileOutline, alpha, isRadByDim, isRadBySize, maxGen)
    quadTree.printQuadTreeNodes(quadTree.quadRoot)

if __name__=='__main__':
    main()