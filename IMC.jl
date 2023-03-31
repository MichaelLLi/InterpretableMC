
@everywhere using DataFrames, JuMP,Gurobi, StatsBase, LinearAlgebra, AmplNLWriter, GAMS, SCIP, Juniper, Ipopt



function MatrixOptIntFull(A,B,k,γ,W=false)
    # Unknown values should be in NaN for A
    # A is the matrix that needs to be imputed, size n x m
    # B is the feature selector, size m x p
    # k is the maximum rank of the resultant matrix
    B=B./sqrt.(sum(B.*B,dims=1))
    n=size(A)[1]
    m=size(A)[2]
    if m != size(B)[1]
        error("Sizes of A and B must match")
    end
    p=size(B)[2]
    Aopt=zeros(n,m)
    M=sum(ismissing.(A))/(n*m)
    if W==false
        W=zeros(n,m)
        for i=1:n
            W[i,:]=ones(m)-ismissing.(A[i,:])
        end
    end
    A[ismissing.(A)].=0
    println("Preprocessing Complete")
    m1 = Model(
        optimizer_with_attributes(
            Gurobi.Optimizer, "OutputFlag" => 1, "LazyConstraints" => 1, "Heuristics" => 0
        )
    )
    # Add variables
    @variable(m1, z[1:p],Bin)
    @variable(m1,t>= 0)
    # Add constraints
    @constraint(m1, sum(z[i] for i=1:p)==k)
    # Set nonlinear objective function with @NLobjective
    
    @objective(m1, Min, t)
    z0=zeros(p)
    samplek = sample(1:p, k, replace = false)
    z0[samplek].=1
    obj0, ∇obj0 = Cutting_plane_full(A,B,W,z0,k,γ,M)
    @constraint(m1, t >= obj0 + dot(∇obj0, z - z0))
    # Outer approximation method for Convex Integer Optimization (CIO)
    function Newcut(cb)
      z_cur = [callback_value(cb, z[i]) for i=1:p]
      obj, ∇obj = Cutting_plane_full(A,B,W,z_cur,k,γ,M)
    
      # add the cut: t >= obj + sum(∇s * (s - s_val))
      offset = sum(∇obj .* z_cur)
      MOI.submit(m1, MOI.LazyConstraint(cb),  @build_constraint(t >= obj + sum(∇obj[j] * z[j] for j=1:p) - offset))
    end
    MOI.set(m1, MOI.LazyConstraintCallback(), Newcut)
    println("Model Setup Complete")
    # Solve the model and get the optimal solutions
    optimize!(m1)
    zopt = value.(z)
    X=B[:,zopt.>0.5]
    println("Model Solved")
    function PopulateA(Xtemp,Atemp)
        return pinv(Xtemp'*Xtemp,1e-7)*(Xtemp'*Atemp)
    end
    Xtemp=Array{Float64}[X[W[i,:].==1,:] for i=1:n]
    Atemp=Array{Float64}[A[i,W[i,:].==1] for i=1:n]
    Uopt = zeros(n,k)
    result=pmap(PopulateA,Xtemp,Atemp)
    for i=1:n
        Uopt[i,:]=result[i]
    end
    return Uopt, X
end

function MatrixOptInt(A,B,k,γ,W=false)
    # Unknown values should be in NaN for A
    # A is the matrix that needs to be imputed, size n x m
    # B is the feature selector, size m x p
    # k is the maximum rank of the resultant matrix
    B=B./sqrt.(sum(B.*B,dims=1))
    n=size(A)[1]
    m=size(A)[2]
    if m != size(B)[1]
        error("Sizes of A and B must match")
    end
    p=size(B)[2]
    Aopt=zeros(n,m)
    M=sum(ismissing.(A))/(n*m)
    if W==false
        W=zeros(n,m)
        for i=1:n
            W[i,:]=ones(m)-ismissing.(A[i,:])
        end
    end
    A[ismissing.(A)].=0
    println("Preprocessing Complete")
    m1 = Model(
        optimizer_with_attributes(
            Gurobi.Optimizer, "OutputFlag" => 1, "LazyConstraints" => 1, "Heuristics" => 0
        )
    )
    # Add variables
    @variable(m1, z[1:p],Bin)
    @variable(m1,t>= 0)
    # Add constraints
    @constraint(m1, sum(z[i] for i=1:p)==k)
    # Set nonlinear objective function with @NLobjective
    
    @objective(m1, Min, t)
    z0=zeros(p)
    samplek = sample(1:p, k, replace = false)
    z0[samplek].=1
    obj0, ∇obj0 = Cutting_plane(A,B,W,z0,k,γ,M)
    @constraint(m1, t >= obj0 + dot(∇obj0, z - z0))
    # Outer approximation method for Convex Integer Optimization (CIO)
    function Newcut(cb)
      z_cur = [callback_value(cb, z[i]) for i=1:p]
      obj, ∇obj = Cutting_plane(A,B,W,z_cur,k,γ,M)
    
      # add the cut: t >= obj + sum(∇s * (s - s_val))
      offset = sum(∇obj .* z_cur)
      MOI.submit(m1, MOI.LazyConstraint(cb),  @build_constraint(t >= obj + sum(∇obj[j] * z[j] for j=1:p) - offset))
    end
    MOI.set(m1, MOI.LazyConstraintCallback(), Newcut)
    println("Model Setup Complete")
    # Solve the model and get the optimal solutions
    optimize!(m1)
    zopt = value.(z)
    X=B[:,zopt.>0.5]
    println("Model Solved")
    function PopulateA(Xtemp,Atemp)
        return pinv(Xtemp'*Xtemp,1e-7)*(Xtemp'*Atemp)
    end
    Xtemp=Array{Float64}[X[W[i,:].==1,:] for i=1:n]
    Atemp=Array{Float64}[A[i,W[i,:].==1] for i=1:n]
    Uopt = zeros(n,k)
    result=pmap(PopulateA,Xtemp,Atemp)
    for i=1:n
        Uopt[i,:]=result[i]
    end
    return Uopt, X
end

function Cutting_plane(A,B,W,z0,k,γ,M)
    p=size(B)[2]
    n=size(A)[1]
    m=convert(Int,size(A)[2])
    ∇obj = zeros(p)
    nsquare = sqrt(n*m)
    nsamples = min(nsquare*log(nsquare)*k,n*m)
    # nnew = n
    # mnew = m
    # nnew = min(n, Int(round(nsamples/ min(100,m))))
    # mnew = min(100, m)
    nnew = min(100, n)
    mnew = min(m, Int(round(nsamples/ min(100,n))))
    function SmallInv(Xrow,Arow)
        return Arow - Xrow * (inv(I / γ + Xrow' * Xrow) * (Xrow' * Arow))
    end
    samplen = sample(1:n, nnew, replace=false)
    Wpar = Array{Int64}[(1:m)[W[samplen[i],:].==1] for i=1:nnew]
    samplem = Array{Int64}[sample(Wpar[i],min(mnew,length(Wpar[i])),replace=false) for i = 1:nnew]
    Xpar = Array{Float64}[B[samplem[i],z0.>0.5] for i=1:nnew]
    Apar = Array{Float64}[A[samplen[i],samplem[i]] for i=1:nnew]
    objpar = pmap(SmallInv,Xpar,Apar)
    obj = @distributed (+) for i=1:nnew
        dot(Apar[i],objpar[i])/(2*mnew*nnew)
    end
    ∇obj=@distributed (+) for i=1:nnew
        -γ*(B[samplem[i],:]'*objpar[i]).^2/(2*mnew*nnew)
    end
    return obj, ∇obj
end

function Cutting_plane_full(A,B,W,z0,k,γ,M)
    p=size(B)[2]
    n=size(A)[1]
    m=convert(Int,size(A)[2])
    ∇obj = zeros(p)
    function SmallInv(Xrow,Arow)
        return Arow - Xrow * (inv(I / γ + Xrow' * Xrow) * (Xrow' * Arow))
    end
    active_columns = [z0[i] > 0.5 for i in 1:length(z0)]
    Wpar = Array{Int64}[(1:m)[W[i,:].==1] for i=1:n]
    Xpar = Array{Float64}[B[Wpar[i],active_columns] for i=1:n]
    Apar = Array{Float64}[A[i,Wpar[i]] for i=1:n]
    objpar = pmap(SmallInv,Xpar,Apar)
    obj = @distributed (+) for i=1:n
        dot(Apar[i],objpar[i])/(2*m*n)
    end
    ∇obj = @distributed (+) for i=1:n
        -γ * (B[Wpar[i],:]' * objpar[i]) .^ 2 /( 2 * m * n)
    end
    return obj, ∇obj
end


n = 20000
m = 1000
p = 50
k = 10
U = rand(n,k)
V = rand( m,k)
Afull = U * V'
Z = rand(m,p-k)

B = hcat(Z,V)
A = copy(Afull)
A = allowmissing(A)
A[rand(n,m).>0.05] .= missing
t1 = time_ns()
Uopt,Vopt=MatrixOptIntDirect(A,B,k,100)
println((time_ns()-t1)/1e9)
mean(abs.(Uopt*Vopt'.-Afull)./abs.(Afull))