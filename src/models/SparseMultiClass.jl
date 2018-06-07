
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SparseMultiClass <: MultiClassGPModel
    @commonfields
    @functionfields
    @multiclassfields
    @multiclassstochasticfields
    @kernelfields
    @multiclass_sparsefields
    function SparseMultiClass(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,KStochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,KSize::Int64=-1,BatchSize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    VerboseLevel::Integer=0)
            Y,y_map,y_class = one_of_K_mapping(y)
            this = new()
            this.ModelType = MultiClassGP
            this.Name = "Sparse MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initMultiClass!(this,Y,y_class,y_map);
            initMultiClassSparse!(this,m,OptimizeIndPoints)
            if Stochastic
                initMultiClassStochastic!(this,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = collect(1:this.nSamples); this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0; this.ρ_s=ones(Float64,this.K)
            end
            initMultiClassVariables!(this,μ_init)
            return this;
    end
end
