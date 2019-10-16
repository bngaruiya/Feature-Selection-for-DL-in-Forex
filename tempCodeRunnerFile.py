for j in keylist[i]:
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + k
                masterFrame[colID] = dictlist[i][j][k]
