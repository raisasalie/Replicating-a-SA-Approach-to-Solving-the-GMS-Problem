rm(list = ls(all=T))
set.seed(2020)
################# DATA ###################
##### 21-unit test system data
data_21_unit <- c(1, 555, 1, 20,
                  2, 555, 27, 48,
                  3, 180, 1, 25,
                  4, 180, 1, 26,
                  5, 640, 27, 48,
                  6, 640, 1, 24,
                  7, 640, 1, 24,
                  8, 555, 27, 47,
                  9, 276, 1, 17,
                  10, 140, 1, 23,
                  11, 90, 1, 26,
                  12, 76, 27, 50,
                  13, 76, 1, 25,
                  14, 94, 1, 23,
                  15, 39, 1, 25,
                  16, 188, 1, 25,
                  17, 58, 27, 52,
                  18, 48, 27, 51,
                  19, 137, 27, 52,
                  20, 469, 27, 49,
                  21, 52, 1, 24
)

data21 <- matrix(data_21_unit, nrow = 21, 
                 ncol = 4, byrow = T)
colnames(data21) <- c("unit", "G", "E", "L")
duration <- c(7,5,2,1,5,3,3,6,10,4,1,3,2,4,
              2,2,1,2,1,4,3)
data21 <- cbind (data21, duration)
# exclusion - max 20 from among all units
KK21 <- list(list(c(1:21), 20))
# manpower
M21 <- list(c(10,10,5,5,5,5,3),
            c(10,10,10,5,5),
            c(15,15),
            c(20),
            c(10,10,10,10,10),
            c(15,15,15),
            c(15,15,15),
            c(10,10,10,5,5,5), 
            c(3,2,2,2,2,2,2,2,2,3),
            c(10,10,5,5),
            c(20),
            c(10,15,15),
            c(15,15),
            c(10,10,10,10),
            c(15,15), 
            c(15,15),
            c(20),
            c(15,15), 
            c(15),
            c(10,10,10,10),
            c(10,10,10))
# make a data frame 
data21 <- as.data.frame(data21)
# load demand - constant throughout 52 week period 
D21 <- rep(4739, 52)
# safety margin
S21 <- 0.2
# penalty weights 
# window, load, crew
W21 <- c(500000, 1, 200000, 0)
# 20 crew members per week
Mj21 <- rep(20,52)
# exclusion - no exclustions
# can choose up to 21 of entire set 
# (created for generalisation)
KK21 <- list(list(c(1:21), 21))

###################################
### 32-unit test system data
# manpower 
M32 <- list(c(7,7), 
            c(7,7),
            c(12,10,10),
            c(12,10,10),
            c(7,7),
            c(7,7),
            c(12,10,10),
            c(12,10,10),
            c(10,10,15),
            c(10,10,15),
            c(15,10,10),
            c(8,10,10,8), 
            c(8,10,10,8),
            c(8,10,10,8),
            c(4,4), c(4,4), c(4,4), c(4,4), c(4,4),
            c(5,15,10,10), c(5,15,10,10),
            c(15,10,10,10,10,5), c(15,10,10,10,10,5),
            c(6,6), c(6,6), c(6,6), c(6,6), c(6,6), c(6,6),
            c(12,12,8,8), c(12,12,8,8),
            c(5,10,15,15,5))

# 25 crew members per week 
Mj32 <- rep(25, 52)

# 32-unit test system data
data_32_unit <- matrix(c(1, 20, 1, 25, 2, 
                         2, 20, 1, 25, 2, 
                         3, 76, 1, 24, 3, 
                         4, 76, 27, 50, 3, 
                         5, 20, 1, 25, 2, 
                         6, 20, 27, 51, 2, 
                         7, 76, 1, 24, 3, 
                         8, 76, 27, 50, 3, 
                         9, 100, 1, 50, 3, 
                         10, 100, 1, 50, 3, 
                         11, 100, 1, 50, 3, 
                         12, 197, 1, 23, 4, 
                         13, 197, 1, 23, 4, 
                         14, 197, 27, 49, 4, 
                         15, 12, 1, 51, 2, 
                         16, 12, 1, 51, 2, 
                         17, 12, 1, 51, 2, 
                         18, 12, 1, 51, 2, 
                         19, 12, 1, 51, 2, 
                         20, 155, 1, 23, 4, 
                         21, 155, 27, 49, 4, 
                         22, 400, 1, 21, 6, 
                         23, 400, 27, 47, 6, 
                         24, 50, 1, 51, 2, 
                         25, 50, 1, 51, 2, 
                         26, 50, 1, 51, 2, 
                         27, 50, 1, 51, 2, 
                         28, 50, 1, 51, 2, 
                         29, 50, 1, 51, 2, 
                         30, 155, 1, 23, 4,  
                         31, 155, 1, 49, 4, 
                         32, 350, 1, 48, 5), nrow=32, ncol=5, byrow=T)
colnames(data_32_unit) <- colnames(data21)
data32 <- as.data.frame(data_32_unit)

# exclusion subsets
KK32 <- list(list(c(1,2,3,4),2),
             list(c(5,6,7,8), 2),
             list(c(9, 10, 11), 1),
             list(c(12, 13, 14), 1),
             list(c(15, 16, 17, 18, 19, 20), 3),
             list(c(24, 25, 26, 27, 28, 29), 3),
             list(c(30, 31, 32), 1))
# demand 
D32 <- matrix(c(1, 2457, 14, 2138, 27, 2152, 40, 2063, 
                2, 2565, 15, 2055, 28, 2326, 41, 2118,        
                3, 2502, 16, 2280, 29, 2283, 42, 2120, 
                4, 2377, 17, 2149, 30, 2508, 43, 2280,
                5, 2508, 18, 2385, 31, 2058, 44, 2511, 
                6, 2396, 19, 2480, 32, 2212, 45, 2522,
                7, 2371, 20, 2508, 33, 2280, 46, 2591,
                8, 2296, 21, 2440, 34, 2078, 47, 2679,
                9, 2109, 22, 2311, 35, 2069, 48, 2537,
                10, 2100, 23, 2565, 36, 2009, 49, 2685,
                11, 2038, 24, 2528, 37, 2223, 50, 2765,
                12, 2072, 25, 2554, 38, 1981, 51, 2850,
                13, 2006, 26, 2454, 39, 2063, 52, 2713), ncol=8, byrow = T)
D32 <- c(D32[,-c(1,3,5,7)])
# Safety margin 
S32 <- 0.15
# penalty weights
W32 <- c(40000, 1, 20000, 20000)

#### FUNCTIONS ##########
### function that updates T
### Van Laarhoven
VLupdateT <- function(Temp, Z, g){
  # input: Ts
  # Z = obj value functions so far
  # g = small number 
  # Output: Ts+1
  
  # sd of obj values
  sigs <- sd(Z)
  
  frac <- (log(1+g)/(3*sigs))*Temp
  
  # Ts+1
  val <- Temp*(1/(1 + frac))
  
  return(val)
}

### geometric
GupdateT <- function(Temp, alpha=0.9){
  newT <- alpha*Temp
  return(newT)
} 

### function to transform data to matrices 
trdata <- function(x, data, M){
  # Input:
  # x = possible solution in xi form,
  # data = data pf test set
  # m = total number of time periods 
  # Ouput: 
  # X = xij maintenance schedule
  # Y = yij schedule
  # M = matrix of mij's
  # G = matrix of gij's
  
  # total units 
  n <- nrow(data)
  
  # set up desired matrices 
  X <- matrix(0, ncol=52, nrow=n)
  Y <- X
  
  # transfrom xi's to X
  for(i in 1:length(x)){
    X[i,x[i]] <- 1
    # use duration to get Y from xi = start
    Y[i, (x[i]:min(52,x[i]+data$duration[i]-1))] <- 1
  }
  
  # init matrix for output
  # Manpower matrix of mij's
  MM <- matrix(0, ncol = 52, nrow = nrow(data))
  # Generating capacity matrix of gij's
  GG <-matrix(data$G, ncol=52, nrow = length(x), byrow = F)
  # populate
  for (i in 1:length(x)){
    # ith row of Y
    yi <- Y[i,]
    
    # populate 
    MM[i,as.logical(yi)] <- M[[i]][1:sum(yi)] 
    GG[i,as.logical(yi)] <- 0
  }
  
  # return both X and Y
  return(list(X=X, Y=Y, M=MM, G=GG))
}

## function to generate ejection chain - FINAL
createEjectionChainList <- function(Unit, E, L, wext, x, duration){
  #Input: 
  # i = The unit at the head of an ejection chain (randomly selected ), 
  # n = the number of units, 
  # e,l = the vectors containing the earliest and latest maintenance starting times for all units, 
  # wext = the maintenance commencement extension parameter, 
  # x = the current solution vector 
  # Output: chain
  
  # init chain
  chain <- matrix(NA, ncol=2, nrow=1)
  
  # Possible times to start unit
  upper <- min(L[Unit] + wext, 52 - duration[Unit] + 1)
  lower <- max(1, E[Unit] - wext)
  possibleTimes <- c(lower:upper)
  # remove x[Unit] if in possibleTimes
  if(x[Unit] %in% possibleTimes){
    possibleTimes <- possibleTimes[-which(possibleTimes==x[Unit])]
  }
  
  # new value for x[Unit]
  newTime <- sample(rep(possibleTimes,2),1)
  
  # set counter 
  counter <- 1
  
  # store 
  chain[counter, ] <- c(Unit, newTime)
  
  # which units to choose from
  remainingUnits <- c(1:length(x))[-Unit]
  
  # Begin loop
  notDone <- TRUE
  while(notDone==T){
    # init potential units to choose from
    potentialUnits <- c()
    for(i in remainingUnits){
      # check which units share time with x[Unit]
      if(x[i]==newTime){
        potentialUnits <- c(potentialUnits, i)
      }
    }
    
    # if there are units that share time with x[Unit]
    if(length(potentialUnits) > 0){
      # set counter for storage
      counter <- counter + 1
      
      # randomly choose unit from potentials
      newUnit <- sample(rep(potentialUnits,2), 1)
      
      # possible starting times for newUnit
      upper <- min(L[newUnit] + wext, 52 - duration[newUnit] + 1)
      lower <- max(E[newUnit] - wext, 1)
      possibleTimes <- c(lower:upper)
      if(x[newUnit] %in% possibleTimes){
        possibleTimes <- possibleTimes[-which(possibleTimes==x[newUnit])]
      }
      
      # randomly select new time 
      newTime <- sample(rep(possibleTimes,2), 1)
      
      # store newUnit, newTime
      chain <- rbind(chain, c(newUnit, newTime))
      
      # remove new unit from remaining units
      remainingUnits <- c(1:length(x))[-newUnit]
      
      # check if returned to first unit
      if(newTime==chain[1,2]){notDone <- FALSE}
    }else{ 
      # if there are no potential units, end
      notDone <- FALSE
    }
  }
  
  return(chain)
}

# function to calc P
checkFeasibilityAndCalculatePenalty <- function(x, data, W, D, Mjs, KK, M){
  # Input: 
  # x = The current solution vector, 
  # data = the problem's full dataset
  # W = vector of weights
  # D = demand 
  # Mjs = vector of overall manpower upperbound
  # KK = set of exclusion subsets
  # M = manpower upperbound list
  # Output: The total penalty term for the current solution
  
  # init penalties for each constraint set
  Pw <- 0
  Pl <- 0
  Pc <- 0
  Pe <- 0
  
  # required matrices
  X <- trdata(x = x, data = data, M=M)$X
  Y <- trdata(x = x, data = data, M=M)$X
  M <- trdata(x = x, data = data, M=M)$M
  G <- trdata(x = x, data = data, M=M)$G
  
  
  # Maintenance window
  for(i in 1:length(x)){
    ei <- data$E[i]
    li <- data$L[i]
    wi <- W[1]
    
    if(!(x[i] %in% (ei:li))){
      if(x[i] < ei){pwi <- ei-x[i]}else{pwi <- x[i] - li}
      # add pwi*weight 
      Pw <- Pw + wi*pwi
    }
  }
  
  
  # Load constraints 
  rjs <- colSums(G) - D
  for (rj in rjs){
    # find penalty for time period j
    plj <- max(-rj, 0)
    # add to total load penalty
    Pl <- Pl + plj
  }
  
  
  # Crew constraints 
  vals <- colSums(M) - Mjs
  for(v in vals){
    pcj <- max(0, v)
    # add to sum
    Pc <- Pc + pcj
  }
  
  
  # Exclusion constraints
  for(k in 1:length(KK)){
    # exclusion subset
    Ik <- unlist(KK[[k]][1])
    # max units allowed
    Kk <- unlist(KK[[k]][2])
    # total units in Ik in operation for each j
    sumyij <- colSums(Y[Ik,])
    
    for (j in 1:52){ 
      # penalise if +ve deviation
      if(sumyij[j] - Kk >0){
        pekj <- sumyij[j] - Kk
        # add to total penalisation sum 
        Pe <- Pe + pekj
      }
    }
  }
  
  # return weighted sum
  return(sum(W*c(Pw, Pl, Pc, Pe)))
}

# function that creates list of classical moves
createClassicalNeighbourhoodList <- function(n, E, L, wext, duration){
  #Input: 
  # n= The number of units, 
  # E, L =  the vectors containing the earliest and latest maintenance 
  # starting times for all units, the maintenance commencement extension
  # parameter
  # duration = vector of maintenance durations
  #Output: 
  # The list of elementary moves that creates the full classical neighbourhood
  
  # init counter
  counter <- 1
  
  # init moves matrix 
  moves <- matrix(NA, ncol=2, nrow=1)
  
  for (i in 1:n){
    for (j in (max(1,E[i]-wext):min(52-duration[i]+1,L[i]+wext))){
      # add to moves 
      moves <- rbind(moves, c(i,j))
      # update counter
      counter <- counter+1
    }
  }
  
  # return moves without first row
  return(moves[-1,])
}

### function that calculates the objective value of a solution
calcObjVal <- function(x, data, D, M){
  # Input: 
  # x = possible solution
  # data = all data
  # Output:
  # objective function value -> sum(rj)^2
  
  g <- data$G
  Y <- trdata(x = x, data = data, M=M)$Y
  
  # sum of Pljj's
  totsum <- 0
  
  for(j in 1:ncol(Y)){
    # second term 
    t2 <- D[j]*(1+S)
    
    # first term 
    t1sum <- 0
    for (i in 1:length(x)){
      gij <- g[i]
      yij <- Y[i,j]
      t1sum <- t1sum + gij*(1-yij)
    }
    
    # rj 
    rj <- max(0, t1sum - t2)
    
    # rj^2
    val <- rj^2
    
    # add to total sum
    totsum <- totsum + val
  }
  
  # return total sum
  return(totsum)
}

### function that generates a random solution
generateRandomSolution <- function(data, wext, M, 
                                   W, D, Mjs, K){
  # Input: The problem's full dataset
  # Output: A random solution vector and it's objective function value
  
  # init vector of solutions 
  x <- c()
  
  for (i in 1:nrow(data)){
    possibleTimes <- c(max(data$E[i] + wext,0):min(52-data$duration[i]+1, 
                                                   data$L[i] + wext))
    # sample from possible staring times 
    xi <- sample(possibleTimes, 1)
    # update vector  
    x <- c(x, xi)  
  }
  
  # calculate penalty 
  P <- checkFeasibilityAndCalculatePenalty(x=x, data=data, M=M, W=W, D=D, 
                                           Mjs = Mjs, KK = K)
  # calculate objective value of x, and add P
  objval <- calcObjVal(x = x, data = data, M=M, D = D)
  objval <- objval + P
  
  # return solution and its oenalised obj val
  return(list(x, objval))
}

### function that transforms x based on ejection chain list
applyChain <- function(x, Chain){\
  # Input: 
  # x = previous solution
  # Chain = output of createEjectionChainList
  # Output:
  # transformed solution x
  newx <- x
  
  for(i in 1:nrow(Chain)){
    # extract unit, time to be changed
    unit <- Chain[i,1]
    time <- Chain[i,2]
    
    newx[unit] <- time 
  }
  return(newx)
}

### function that generates a random solution
initialTemperature <- function(x, xObj, data, M, D, 
                               chi0, W, Mjs, K, rwlength=20){
  # Input: The initial solution vector, 
  # xObj =  the initial objective function value, 
  # data = the problem’s full dataset
  # rwlength = length of random walk
  # Output: Two initial temperatures 
  # calculated using the average increase method
  # using the standard deviation method 
  current <- x
  currentObj <- xObj
  
  # init storage of increases and values
  increases <- c()
  values <- c()
  
  j <- 0
  
  for(i in 1:rwlength){
    # store prev obj function value
    prevObj <- currentObj
    # randomly select a unit
    unit <- sample(1:length(x),1)
    # gen an ejection chain for unit
    chain <- createEjectionChainList(Unit = unit, E = data$E, L = data$L, 
                                     x = current, wext = 2, duration = data$duration)
    # apply chain to current x
    # reset current x 
    current <- applyChain(x = current, Chain = chain)
    # calculate penalty of current x
    P <- checkFeasibilityAndCalculatePenalty(x = current, data = data, D = D, 
                                             M=M, W=W, Mjs = Mjs, KK=K)
    # calculate opbj function value of current x
    currentObj <- calcObjVal(x = current, data = data, D = D, M = M) + P
    # calculate change in obj function from last run
    DeltaE <- currentObj - prevObj
    # if solution got worse 
    if(DeltaE > 0){
      # update j
      j <- j+1
      # store increase
      increases <- c(increases, DeltaE)
    }
    # store obj value
    values <- c(values, currentObj)
  }
  
  # ave increase in temperature
  avgIncTemperature <- -mean(increases)/log(chi0)
  # sd of objective function
  stdDevTemperature <- sd(values)
  
  # return two temp options
  return(list(av = avgIncTemperature, sd =stdDevTemperature))
}
###############################################
##### GMS local search heuristic ######
runSearchHeur <- function(incumbent, incumbentObj, 
                          data, M, D, W, Mj, K){
  # Input: 
  # xinc = The incumbent solution vector, 
  # objinc = the incumbent objective function value, 
  # data = the problem’s full dataset
  #Output: 
  # The possibly improved incumbent solution vector 
  # and corresponding objective function value
  
  # set incumbent as current
  current <- incumbent
  currentObj <- incumbentObj
  
  # set improved indicator
  improved <- TRUE
  
  # create neighbourhood list of current solution
  moves <-createClassicalNeighbourhoodList(n = length(current), 
                                           E = data$E, L=data$L, 
                                           wext = 2, duration = data$duration)
  
  while(improved == T){
    # init besst neighbour storage
    bestNeighbour <- c()
    # set arbitrarily large obj function value
    bestNeighbourObj <- 10^20
    
    for (i in 1:nrow(moves)){
      neighbour <- current
      # rows to choose from (corresponding to unit i)
      row <- matrix(moves[i,], nrow=1)
      # apply move to neighbour to create new neighbour
      neighbour <- applyChain(x = neighbour, Chain = row)
      # calculate penalty of new neighbour
      P <- checkFeasibilityAndCalculatePenalty(x = neighbour, data = data, 
                                               W = W, D = D, Mjs = Mj, KK = K, M = M)
      # calculate obj value of new neighbour
      neighbourObj <- calcObjVal(x = neighbour, data = data, D = D, M = M)
      neighbourObj <- neighbourObj + P
      
      # if new neighbour's obj value better than best so far
      if(neighbourObj < bestNeighbourObj){
        # set new neighbour as best neighbour 
        bestNeighbour <- neighbour
        bestNeighbourObj <- neighbourObj
      }
    }
    
    # if final best neighbour is better than current incumbent solution
    if(bestNeighbourObj < incumbentObj){
      # set best neighbour as new incumbent solution
      incumbent <- bestNeighbour
      incumbentObj <- bestNeighbourObj
    }else{ # if best neighbour is not better than current incumbent solution
      # incumbent solution is not improved by any neighbours generated 
      improved <- FALSE
    }
  }
  
  return(list(sol=incumbent, obj=incumbentObj))
}

# function to generate a better random solution
generateGoodRandomSolution <- function(no, data, M, W, 
                                       D, Mjs, K){
  # Input: 
  # no = The number of solutions to compare, 
  # data = the problem’s full dataset 
  # Output: 
  # A good random solution vector, the objective function value
  
  # init best obj value as arbitrarily large number 
  bestObj <- 10^20
  
  for (i in 1:no){
    # generate a random solution and obj value
    rand <- generateRandomSolution(data = data, wext = 2, M = M, W = W, 
                                   D = D, Mjs = Mjs, K = K)
    solution <- rand[[1]]
    solutionObj <- rand[[2]]
    # apply local search heuristic to improve random solution
    heur <- runSearchHeur(data = data, M = M, D = D, W = W, Mj = Mjs, 
                          K = K, incumbent = solution, incumbentObj = solutionObj)
    solution <- heur$sol
    solutionObj <- heur$obj
    
    # if new solution is better than best so far
    if(solutionObj < bestObj){
      # store solution and its obj value
      best <- solution
      bestObj <- solutionObj
    }
  }
  
  # return the best solutions from loop
  return(list(sol=best, obj=bestObj))
}

#### HYBRIDISATION ###
runHybrid <- function(data, M, W, D, Mj, K, S, delta, 
                      Tmin, omega_fr, initTemp, initSol, initObj){
  #Input: 
    # A power system scenario for which to solve the generator 
    # maintenance scheduling problem
    # initTemp, initSol previously generated
  #Output: 
    # The best maintenance schedule found
  
  
  # set seed
  set.seed(2020)
  
  # init storage of 
  
  # starting solution and obj value
  currentObj <- initObj
  
  # initial temps using both methods
  inittemp <- initTemp
  avgT0 <- inittemp[1]
  sdT0 <- inittemp[2]
  
  # use avgT0
  # temp=T
  Temp <- avgT0
  
  # init incumbent solution
  incumbent <- current
  incumbentObj <- currentObj
  
  # init count of solutions not accepted 
  notAcceptCounter <- 0
  
  # init vector of all obj function values
  obj_all <- c()
  obj_allInc <- c()
  
  # init vector of storage of temps
  Temps <- c()
  
  # while termination criteria are not met
  while((Temp > Tmin) & (notAcceptCounter < omega_fr)){
    # set count of no. accepted, attempted
    numberAccept <- 0
    numberAttempt <- 0
    # init accepted indicator
    accepted <- FALSE
    
    while((numberAccept < 12*nrow(data)) & (numberAttempt < 100*nrow(data))){
      # update attempt count
      numberAttempt <- numberAttempt + 1
      print(numberAttempt)
      # init neighbour
      neighbour <- current
      # randomly select a unit 
      unit <- sample(1:nrow(data), 1)
      # create ejection chain for neighboour starting at this unit
      chain <- createEjectionChainList(Unit = unit, E = data$E, L = data$L, 
                                       wext = 2, x = neighbour, duration = data$duration)
      # apply chain on neighbour to get new neighbour
      neighbour <- applyChain(x = neighbour, Chain = chain)
      # calculate feasibility penalty of new neighbour
      P <- checkFeasibilityAndCalculatePenalty(x = neighbour, data = data, W = W, D = D, 
                                               Mjs = Mj, KK = K, M = M)
      # calculate objective function value of new neighbour
      neighbourObj <- calcObjVal(x = neighbour, data = data, D = D, M = M)
      # add feasibility penalty
      neighbourObj <- neighbourObj + P
      # calculate change in objective value between current and neighbour
      DeltaE <- neighbourObj - currentObj
      
      # if neighbour is better
      if(DeltaE <= 0){
        # store neighbour and its objective value
        current <- neighbour
        currentObj <- neighbourObj
        # store obj function value
        obj_all <- c(obj_all, currentObj)
        
        # update # accepted
        numberAccept <- numberAccept + 1
        # set accepted indicator 
        accepted <- TRUE
        
        # if current obj value is better that incumbent's
        if(currentObj < incumbentObj){ # line 26
          # replace incumbent solution and obj value
          incumbent <- current
          incumbentObj <- currentObj
          # apply search heuristic to incumbent solution
          heur <- runSearchHeur(incumbent = incumbent, incumbentObj = incumbentObj, 
                                data = data, M = M, D = D, W = W, Mj = Mj, K = K)
          incumbent <- heur$sol
          incumbentObj <- heur$obj
          obj_allInc <- c(obj_allInc, incumbentObj)
        }
      }else{ # if neighbour is worse than current 
        # if acceptance conditions are met
        if(runif(1) < exp(-DeltaE/Temp)){
          # accept neighboour
          # set current sol and obj value
          current <- neighbour
          currentObj <- neighbourObj
          # store obj function value
          obj_all <- c(obj_all, currentObj)
          
          # update # accepted 
          numberAccept <- numberAccept + 1
          #print(numberAccept)
          # update accepted counter
          accepted <- TRUE
        }
      }
    }
    
    if(accepted == TRUE){
      notAcceptCounter <- 0
    }else{
      notAcceptCounter <- notAcceptCounter + 1
    }
    
    # update temperature - geometric updating
    Temp <- GupdateT(Temp = Temp)
    #Temp <- VLupdateT(Temp = Temp, Z = obj_all, g = delta)
    
    # add to storage 
    Temps <- c(Temps, Temp)
  }
  
  return(list(obj=obj_all, sol=current, inc=incumbent, incObj = incumbentObj, 
              incObj_all=obj_allInc, Temps=Temps))
}

### function to check feasibility
is.feasible <- function(x, data, W, D, Mjs, KK, M){
  P <- checkFeasibilityAndCalculatePenalty(x = x, data = data, W = W, 
                                      D = D, Mjs = Mjs, KK = KK, M = M)
  # check if feasible
  feas <- P==0
  
  return(feas)
}

####################################
# get temps and good solutions beforehand
# 32-unit
S <- S32
TEMPS32 <- c()
INITS32 <- c()
OBJS32 <- c()
# generate init conditions
set.seed(2020)
for (i in 1:50){
  # init sol
  soln <- generateGoodRandomSolution(no = 2, data = data32, 
                                     M = M32, W = W32, D = D32, Mjs=Mj32, K=KK32)
  initi <- matrix(soln$sol, nrow=1)
  initiObj <- soln$obj
  # store
  INITS32 <- rbind(INITS32, initi)
  OBJS32 <- c(OBJS32, initiObj)
  
  # init temp
  tmp <- initialTemperature(x=initi, xObj = soln$obj, data = data32, M = M32, 
                            D = D32, chi0 = 0.5, W = W32, Mjs = Mj32, K = KK32, 
                            rwlength = 100)
  # store
  bothTemps <- tmp
  TEMPS32 <- rbind(TEMPS32, matrix(unlist(bothTemps), nrow=1))
}
# save.image("temps32FINAL.RData")

# 21-unit
S <- S21
TEMPS21 <- c()
INITS21 <- c()
OBJS21 <- c()
# generate init conditions
set.seed(2020)
for (i in 1:50){
  # init sol
  soln <- generateGoodRandomSolution(no = 2, data = data21, 
                                     M = M21, W = W21, D = D21, Mjs=Mj21, K=KK21)
  initi <- matrix(soln$sol, nrow=1)
  initiObj <- soln$obj
  # store
  INITS21 <- rbind(INITS21, initi)
  OBJS21 <- c(OBJS21, initiObj)
  
  # init temp
  tmp <- initialTemperature(x=initi, xObj = soln$obj, data = data21, M = M21, 
                            D = D21, chi0 = 0.5, W = W21, Mjs = Mj21, K = KK21, 
                            rwlength = 100)
  # store
  bothTemps <- tmp
  TEMPS21 <- rbind(TEMPS21, matrix(unlist(bothTemps), nrow=1))
}
library(beepr)
beep()
# save 
# save.image("allinitFINAL.Rdata")
# load(file = "allinitFINAL.Rdata")

####### 32-unit run ##########
# set up parallel
library(doParallel)
cl <- makeCluster(max(1,detectCores() - 1))
registerDoParallel(cl)

########## LOOP 32 ################
# storage of all results
set.seed(2020)
S <- S32

RUNS32 <-foreach(R = 1:3) %dopar% {
        print(paste("run" , R-1, "complete", sep=" "))
        runHybrid(data = data32, M = M32, W = W32, D = D32, 
                   Mj = Mj32, K = KK32, S = S32, delta = 0.35, 
                  Tmin = 1, omega_fr = 100, initTemp = TEMPS32[R,],
                  initSol = INITS32[R,], initObj = OBJS32[R])
}
beep()
stopCluster(cl)
# save.image(file = "32RunFINALDAY.RData")

# check feasibility 
F32 <- c()
for (i in 1:length(RUNS32)){
  F32 <- c(F32, is.feasible(RUNS32[[i]]$sol, data = data32, W = W32, D = D32, 
              Mjs = Mj32, KK = KK32, M = M32))
}
F32 # feasible!

#################################################
####################################
####### 21-unit run ##########
cl <- makeCluster(max(1,detectCores() - 1))
registerDoParallel(cl)

########## LOOP 21-unit ################
# storage of all results
set.seed(2020)
S <- S21

RUNS21 <-foreach(R = 1:3) %dopar% {
  print(paste("run" , R-1, "complete", sep=" "))
  runHybrid(data = data21, M = M21, W = W21, D = D21, 
            Mj = Mj21, K = KK21, S = S21, delta = 0.35, 
            Tmin = 1, omega_fr = 100, initTemp = TEMPS21[R,],
            initSol = INITS21[R,], initObj = OBJS21[R])
}
beep()
stopCluster(cl)

# check feasibility 
F21 <- c()
for (i in 1:length(RUNS21)){
  F21 <- c(F21, is.feasible(RUNS21[[i]]$sol, data = data21, W = W21, D = D21, 
                            Mjs = Mj21, KK = KK21, M = M21))
}
F21 # feasible!
#save.image(file = "ALLRUNSFINAL.RData")
#load(file = "ALLRUNSFINAL.RData")
#################################################
##### PLOT RESULTS ######
ALLRUNS <- list(RUNS32, RUNS21)
names(ALLRUNS) <- c("32", "21")

#######
for(test in 1:2){
  pdf(paste("plots", names(ALLRUNS), ".pdf", sep = "")[test],
      width = 15, height = 15, compress = F)
  # set up plot matrix
  par(mfrow=c(3,3))
  # 32 or 21
  runs <- ALLRUNS[[test]]
  for (i in 1:length(runs)){
    # choose run
    thisrun <- runs[[i]]
    ## PLOT
    # obj value 
    plot(thisrun$obj, type = "l", ylim = c(0, 4e+07),
           #ylim=c(0, 1e+08), for 21-unit
         lwd = 1, col="cadetblue4", 
         ylab="Candidate solution evaluation function value")
    # inc obj value
    plot(thisrun$incObj_all, type = "l", 
         lwd = 2, col = "cadetblue3", 
         ylab="Incumbent solution evaluation function value")
    # temp
    plot(thisrun$Temps, type="l", 
         lwd = 2, col = "cadetblue2", ylab="Temperature")
  }
  # save pdf
  #dev.off()
}

######## SUMMARY TABLES ########
## 32 UNIT
restable32 <- matrix(NA, ncol = 34, nrow=6)
# first row for names
restable32[,1] <- rep(c("Cand.", "Inc."),3)

for(i in 1:3){
  # solutions
  restable32[2*i-1,2:33] <- RUNS32[[i]]$sol
  restable32[2*i, 2:33] <- RUNS32[[i]]$inc
  
  # evaluation function values
  restable32[2*i-1,34] <- calcObjVal(x = RUNS32[[i]]$sol, 
                                     data = data32, D = D32, M = M32)
  restable32[2*i,34] <- RUNS32[[i]]$incObj
}

#library(xtable)
#xtable(t(restable32))

## 21 Unit
restable21 <- matrix(NA, ncol =23, nrow=6)
# first row for names
restable21[,1] <- rep(c("Cand.", "Inc."),3)

for(i in 1:3){
  # solutions
  restable21[2*i-1,2:22] <- RUNS21[[i]]$sol
  restable21[2*i, 2:22] <- RUNS21[[i]]$inc
  
  # evaluation function values
  restable21[2*i-1,23] <- calcObjVal(x = RUNS21[[i]]$sol, 
                                     data = data21, D = D21, 
                                     M = M21)
  restable21[2*i,23] <- calcObjVal(x = RUNS21[[i]]$inc, 
                                   data = data21, D = D21, M = M21)
}

xtable(t(restable21), 
       caption = "Results from three runs of 21-unit test system", 
       digits = 0)
########## END 
















