using LinearAlgebra
using Statistics
using Distances
using ScikitLearn: fit!
using ScikitLearn.Kernels: RBF




struct TCA
    dim::Int
    lambda::Float64
    gamma::Float64
    kernel::Function
end

function TCA(dim=30, lambda=1.0, gamma=1.0)
    kernel = (x) -> 0.5 * RBF(gamma, "fixed")(x)
    return TCA(dim, lambda, gamma, kernel)
end

function fit(tca::TCA, Xs, Xt)
    X = vcat(Xs, Xt)
    K = tca.kernel(X)
    ns, nt = size(Xs, 1), size(Xt, 1)

    if tca.dim > (ns + nt)
        error("The maximum number of dimensions should be smaller than $(ns + nt)")
    end

    e = vcat((1 / ns) * ones((ns, 1)), (-1 / nt) * ones((nt, 1)))
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
