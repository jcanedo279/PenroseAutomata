def editDist(source, target):
    ## State 0: A len(target)xlen(source) matrix is created to memoize all the edit distances between every substring to source and target
    mem = [[0 for i in range(len(target)+1)] for j in range(len(source)+1)]
    ## State 1.i: For every index of the source string
    for i in range(len(source)+1):
        ## State 1.i.j: For every index of the target string
        for j in range(len(target)+1):
            ## If i or j equal 0, then the edit distance is j or i respectively (the edit distance to add the opposite string)
            if i == 0:
                mem[i][j] = j
            elif j == 0:
                mem[i][j] = i
            ## If source[i-1]==target[j-1] then the edi distance to transform source[:i]->target[:j] == source[:i-1]->target[:j-1]
            elif source[i-1] == target[j-1]:
                mem[i][j] = mem[i-1][j-1]
            ## If none of the above conditions are met, then the edit distance is one plus the minimum edit distance of any smaller or equivalent substring
            ## of either source or target
            else:
                mem[i][j] = 1 + min(mem[i-1][j], mem[i][j-1], mem[i-1][j-1])
    ## State 2: The edit distance between source[:len(source)+1]->target[:len(target)+1] is returned
    return mem[len(target)][len(source)]

print(editDist('Hello', 'Rello')) ## prints 1
print(editDist('Hello', "HellR")) ## prints 1
print(editDist('Hello', 'RellR')) ## prints 2
print(editDist('Hello', 'RRRRR')) ## prints 5