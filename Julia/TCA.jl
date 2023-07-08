using LinearAlgebra
using Statistics
using Distances
using ScikitLearn: fit

struct TCAclass
    dim::Int
    lambda::Float64
    gamma::Float64

    function TCAclass(; dim::Int=30, lambda::Float64=1.0, gamma::Float64=1.0)
        new(dim, lambda, gamma)
    end    
    
    function fit(tca::TCAclass, Xs, Xt)
        X = vcat(Xs, Xt)
        K = rbf_kernel(X, X, tca.gamma)
        ns, nt = size(Xs, 1), size(Xt, 1)

        if tca.dim > (ns + nt)
            error("The maximum number of dimensions should be smaller than $(ns + nt)")
        end

        e = vcat((1 / ns) * ones(ns), (-1 / nt) * ones(nt))
        L = e * e'

        n, _ = size(X)
        H = Matrix(I, n, n) - (1 / n) * ones(n, n)
        matrix = (K * L * K + tca.lambda * Matrix(I, n, n)) * K * H * K'

        w, V = eigen(matrix)
        w, V = real.(w), real.(V)

        ind = sortperm(w)
        A = V[:, ind[1:tca.dim]]

        Z = K * A
        Xs_new, Xt_new = Z[1:ns, :], Z[ns+1:end, :]
        
        return Xs_new, Xt_new
    end
end

function rbf_kernel(A, B, gamma)
    dist_matrix = pairwise(SqEuclidean(), A', B')
    K = exp.(-gamma * dist_matrix)
    return K
end

