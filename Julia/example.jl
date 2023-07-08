using DataFrames
using CSV
using DecisionTree
using Statistics
using PyPlot
using Random
include("TCA.jl")


# Set the random seed
Random.seed!(0)

# source domain training data
Xs_y = DataFrame(CSV.File("Xs_y.csv"))
Xs = Matrix(Xs_y[:, 1:end-1])
ys = convert(Vector, Xs_y[:, end])
# target domain training data
Xt_y = DataFrame(CSV.File("Xt_y.csv"))
Xt = Matrix(Xt_y[:, 1:end-1])
yt = convert(Vector, Xt_y[:, end])
# test data
test = DataFrame(CSV.File("Xtest.csv"))
Xtest = Matrix(test[:, 1:end-1])
ytest = convert(Vector, test[:, end])

reg = DecisionTreeRegressor(max_depth=5)
DecisionTree.fit!(reg, Xt, yt)
################################################################
# case one
# training on target domain data (Xt_y.csv) and test on testing dataset (Xtest.csv)
pre_label = DecisionTree.predict(reg, Xt)
rate = sqrt(mean((pre_label .- yt) .^ 2))
println("Without transfer, we have misclassify rate $rate")

# case two
# training on mapped space (4-d) with source (Xs_y.csv) and target domain data (Xt_y.csv), test on testing dataset (Xtest.csv)
RMSE_list = [rate]
for j in 1:20
    model = TCA.TCA(dim=20-j, lambda=0.3, gamma=1.0)
    Train_X = vcat(Xs, Xt)
    Xs_new, Xt_new = TCA.fit(model, Train_X, Xtest)
    Train_y = vcat(ys, yt)
    transfer_pre_label = DecisionTree.predict(reg, Xs_new)
    transfer_rate = sqrt(mean((transfer_pre_label .- ytest) .^ 2))
    push!(RMSE_list, transfer_rate)
    println("$j dimension")
    println("With transfer, we have misclassify rate $transfer_rate\n")
end

# plot the result
x = length(RMSE_list)-1:-1:1
fig, ax = plt.subplots()
plt.plot(x, RMSE_list[2:end], marker="o", linestyle="--", color="blue", label="with TCA transfer")
ax.axhline(y=RMSE_list[1], color="green", linestyle="--", label="without TCA transfer")
ax.set_xticks(x)

plt.xlabel("TCA dimensions")
plt.ylabel("RMSE on test data")

ax.legend()
plt.savefig("iteration number.png", bbox_inches="tight", dpi=600)
plt.savefig("iteration number.svg", bbox_inches="tight", dpi=600)
plt.show()
