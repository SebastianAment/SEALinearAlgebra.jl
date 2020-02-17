module SEALinearAlgebra


include("./MyLazyArrays.jl")
include("./MyFactorizations.jl")
include("./MyCholesky.jl")
include("./MyWoodburyMatrices.jl")
include("./CovarianceMatrices.jl")
include("./KroneckerProducts.jl")
# include("./RandomizedLinearAlgebra.jl") This seems to be broken?
include("./LinearAlgebraExtensions.jl")

end # module
