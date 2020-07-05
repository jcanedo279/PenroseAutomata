import math

class MultigridCell:
    def __init__(self, dim, r, s, a, b, g_r, g_s, tileType):
        if r==0 and s==0:
            return None
        self.scalingFactor = 1
        ## grid numbers s.t: 0<=r<s<multiGridDim
        self.dim = dim
        self.r = r
        self.s = s
		
		# a is the hyperplane index of grid r
		# b is the hyperplane index of grid s
        self.a = a
        self.b = b

        self.g_r = g_r
        self.g_s = g_s
		
        makeEqn = False
        if makeEqn:
            self.r_eqn = self.getHyperplaneEqn(r, a, g_r)
            self.s_eqn = self.getHyperplaneEqn(s, b, g_s)

        self.tileType = tileType

    def setVertices(self, vertices):
        self.vertices = vertices
        
        # p is the point (kept here for efficiency)
        self.p = self.getVertexPointCord()
        self.p_x = self.p[0]
        self.p_y = self.p[1]

    def setVal(self, val):
        self.val = val

    def setStability(self, isStable):
        self.isStable = isStable
    
    def getHyperplaneEqn(self, i, t, g_i):
		# Calculate grid angles
        i_ang = i*(math.pi)*(2/5)
        cos_i_ang = math.cos(i_ang)
        sin_i_ang = math.sin(i_ang)
		# Calculate y-intercept
        interceptConstant = t + 0.5 - g_i
        hyperplaneIntercept = interceptConstant / cos_i_ang
		# Calculate slope
        hyperplaneSlope = -(sin_i_ang / cos_i_ang)
		# hyperplaneEqn = [slope, y-intercept]
        hyperplaneEqn = [hyperplaneSlope, hyperplaneIntercept]
        return hyperplaneEqn

    def getPointCord(self, r, s, a, b):
        sNormVect = self.genNormVector(self.s)
        rNormVect = self.genNormVector(self.r)

        x_num = a*sNormVect[1] - b*rNormVect[1]
        x_num+= 0.5*sNormVect[1] - 0.5*rNormVect[1]
        x_num+= self.g_s*rNormVect[1] - self.g_r*sNormVect[1]
        x_den = rNormVect[0]*sNormVect[1] - sNormVect[0]*rNormVect[1]
        y_num = a*sNormVect[0] - b*rNormVect[0]
        y_num+= 0.5*sNormVect[0] - 0.5*rNormVect[0]
        y_num+= self.g_s*rNormVect[0] - self.g_r*sNormVect[0]
        y_den = rNormVect[1]*sNormVect[0] - sNormVect[1]*rNormVect[0]

        if x_den==0:
            x_rs = x_num/0.00001
        else:
            x_rs = x_num / x_den
        if y_den==0:
            y_rs = x_num/0.00001
        else:
            y_rs = y_num / y_den

        return (x_rs, y_rs)
    
    def getVertexPointCord(self):
        x_rs = sum([vert[0] for vert in self.vertices])
        y_rs = sum([vert[1] for vert in self.vertices])
        return (x_rs, y_rs)

    def genNormVector(self, i):
        normVectorConstant = 2*(math.pi)*(i/self.dim)
        e_x = math.cos(normVectorConstant)
        e_y = math.sin(normVectorConstant)
        normVector = (self.scalingFactor*e_x, self.scalingFactor*e_y)
        return normVector

    def setColor(self, color):
        self.color = color
