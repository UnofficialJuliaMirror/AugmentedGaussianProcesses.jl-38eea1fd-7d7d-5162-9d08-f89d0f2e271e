"""Structure for the variational heteroscedastic gaussian process regression"""

mutable struct BatchHGP <: FullBatchModel
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    α::Float64
    λ::Vector{Float64}
    c::Vector{Float64}
    γ::Vector{Float64}
    θ::Vector{Float64}
    kernel_g::Kernel
    K_g::Symmetric{Float64,Matrix{Float64}}
    invK_g::Symmetric{Float64,Matrix{Float64}}
    μ_0::Vector{Float64}
    μ_g::Vector{Float64}
    Σ_g::Symmetric{Float64,Matrix{Float64}}
    """BatchStudentT Constructor"""
    function BatchHGP(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),
                                    nEpochs::Integer = 200,
                                    kernel=0,noise::Real=1e-3,AutotuningFrequency::Integer=1,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],verbose::Integer=0,α::Real=5.0,kernel_g=0,μ_0=0)
            this = new()
            this.ModelType = HGP
            this.Name = "Non Sparse GP Regression with Heteroscedastic noise"
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            this.α = α
            this.λ = zeros(Float64,this.nSamples)
            this.c = zero(this.λ)
            this.γ = zero(this.λ)
            this.θ = zero(this.λ)
            this.kernel_g = deepcopy(kernel_g)
            this.K_g = Symmetric(kernelmatrix(this.X,this.kernel_g)+getvariance(kernel_g)*jittering*I)
            this.invK_g = inv(this.K_g)
            this.μ_0 = μ_0*ones(this.nSamples)
            this.μ_g = copy(this.μ_0)
            this.Σ_g = copy(this.K_g)
            return this;
    end
    """Empty constructor for loading models"""
    function BatchHGP()
        this = new()
        this.ModelType = HGP
        this.Name = "Student T Gaussian Process Regression"
        initFunctions!(this)
        return this;
    end
end
