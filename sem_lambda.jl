
using Pkg,LinearAlgebra, SparseArrays, Random,Distributions;

y=[1,2,3]
function get_Y(y)
    m=length(y)#number of rows
    n=m*(m-1) #number of columns
    res = zeros(m,n) #set the result
    start = 1
    for i=1:length(y)
        row = copy(y)
        row = deleteat!(row, i)
        res[i,start:start+m-2]=row
        start = start+m-1
    end
    return res
end
get_Y(y)

#fake data
y1=[1,2,3]
y2=[4,5,6]
Y1=get_Y(y1)
Y2=get_Y(y2)
Y=[Y1,Y2]
R=[1 0 0;0 2 0; 0 0 3]
w1=[6,7,8]
w2=[6,3,4]
W=[w1,w2]
t_square = 3
lambda0 = 4
get_Y(y1)

function get_para(Y,R,t_square,lambda0)
    m = size(Y[1])[1]#number of rows
    n=m*(m-1) #number of columns
    first = zeros(n,n)
    second=zeros(n,1)
    for i=1:length(Y)
        first += Y[i]'*inv(R)*Y[i]
        second += Y[i]'*inv(R)*W[i]
    end
    first += Diagonal(repeat([t_square],n))
    second += repeat([lambda0*t_square],n)
    mu = vec(inv(first)*second)
    return mu,Symmetric(inv(first))
end


#MCMC
mu,var = get_para(Y,R,t_square,lambda0)
rand(MvNormal(mu,Symmetric(var)))

y = [1,2,3]
function get_my_Y(y)
    m=length(y)#number of rows
    n::Int64=(m*(m-1))/2 #number of columns
    res = zeros(m,n) #set the result
    start = 1
    for i = 1:length(y)-1
        row = y[2:i+1]
        res[i+1,start:start + i - 1] = row
        start = start + i
    end
    return res
end
get_my_Y(y)

#Making fake data
y1=[1,2,3]
y2=[4,5,6]
Y1=get_my_Y(y1)
Y2=get_my_Y(y2)
Y=[Y1,Y2]
R=[1 0 0;0 2 0; 0 0 3]
w1=[6,7,8]
w2=[6,3,4]
W=[w1,w2]
t_square = 3
lambda0 = 4

function get_para(Y,R,t_square,lambda0,W)
    m = size(Y[1])[1]#number of rows
    n::Int64=(m*(m-1))/2 #number of columns
    first = zeros(n,n)
    second=zeros(n,1)
    for i=1:length(Y)
        first += Y[i]'*inv(R)*Y[i]
        second += Y[i]'*inv(R)*W[i]
    end
    first += Diagonal(repeat([t_square],n))
    second += repeat([lambda0*t_square],n)    
    mu = vec(inv(first)*second)
    return mu,Symmetric(inv(first))
end

@time mu,var = get_para(Y,R,t_square,lambda0,W)
#rand(MvNormal(mu,Symmetric(var)))

mu

var

res = randn(3)
res = L.U*res + mu
function sampling(mu,var,n)
    res = randn(n)
    L = cholesky(var)
    res = L.L*res + mu
    return res
end
sampling(mu,var,3)

#Making fake data
y1=[1,2,3]
y2=[4,5,6]
Y1=get_my_Y(y1)
Y2=get_my_Y(y2)
Y=[Y1,Y2]
R=[1 0 0;0 2 0; 0 0 3]
w1=[6 7 8]
w2=[6 3 4]
W=[w1 w2]
t_square = 3
lambda0 = 4
number=2


#Y is the data for all individuals

#unlist function will tranfrom y from a list to a big matrix
function unlist(Y)
    res = Y[1]
    for i = 2:length(Y)
        res = vcat(res,Y[i])
    end
    return res
end

unlist(Y)

function get_para(Y,R,W,t_square,lambda0,number)
    m = size(Y[1])[1]#number of rows
    n::Int64=(m*(m-1))/2 #number of columns
    Y = unlist(Y) #Set big Y matrix
    
    #Define sparse matrix
    I = collect(1:number*m); J=collect(1:number*m);V = repeat(diag(R),number)
    R = sparse(I,J,V)
    
    #formula calculation
    first = Y'*inv(Matrix(R))*Y + Diagonal(ones(m,m))
    second = Y'*inv(Matrix(R))*transpose(vcat(W)) + ones(m,1)*lambda0*t_square
    mu = vec(inv(first)*second)
    return mu,Symmetric(inv(first))
end



a, b = get_para(Y,R,W,t_square,lambda0,number)

#Y is a data for single persion
y1 = [1,2,3]
function get_sparse_Y(Y)
    m = size(Y)[1] #number of rows
    n::Int64=(m*(m-1))/2  #number of columns
    col_index = collect(1:n)
    row_index = [];value = []
    for i = 2:m
        current = fill(i,i-1) #get a element i with i-1 time
        row_index = vcat(row_index,current)
        current_value = Y[2:i]
        value = vcat(value,current_value)
    end
    
    #Return the sparse matrix
    row_index = convert(Array{Int64,1}, row_index) #change Any type
    value = convert(Array{Float64,1}, value) #Change Any type

    res = sparse(row_index,col_index,value)
end

get_sparse_Y(y1)

#Fake data
y1=[1,2,3]
y2=[4,5,6]
ys1=get_sparse_Y(y1)
ys2=get_sparse_Y(y2)
YS = [ys1,ys2]
R=[1 0 0;0 2 0; 0 0 3]
w1=[6,7,8]
w2=[6,3,4]
W=[w1,w2]
t_square = 3
lambda0 = 4
m = 3
number = 10000

YS = repeat([get_sparse_Y(rand(3))],10000)
YS = unlist(YS)
#W=unlist(W)
W = rand(30000)

function get_sparse_para(YS,R,W,t_square,lambda0,number,m)
    
    #Define residual matrix
    value = diag(R).^(-1)
    RS = kron(sparse(I,number,number),sparse(1:m,1:m,value))
    
    #formula calculation
    YS_RS_product = YS'*RS
    first = YS_RS_product*YS + t_square*sparse(I,m,m)
    second = YS_RS_product*W + ones(m,1)*lambda0*t_square
    
    #Return the result
    first = inv(Matrix(first))
    mu = vec(first*second)
    var = Symmetric(first)
    
    #Take sample
    res = rand(MvNormal(mu,var))
    return res
end

#a,b = get_sparse_para(YS,R,W,t_square,lambda0,number,3)#
@time lambda= get_sparse_para(YS,R,W,t_square,lambda0,number,m)

y1 = rand(1000);y2 = rand(1000);y3=rand(1000)
BigY = [y1;y2;y3]
Lambda = [1 0 0; lambda[1] 1 0;lambda[2] lambda[3] 1]

function Modify_Y(Lambda,Y,n_individuals)
    Identy = Matrix{Int32}(I,n_individuals,n_individuals)
    res = kron(Identy,Lambda)*BigY
    return res
end

Modify_Y(Lambda,BigY,1000)
